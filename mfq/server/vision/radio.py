"""RADIO (C-RADIO) image preprocessing for cropped-position-embedding towers.

The tower consumes one row-major normalized patch per pixel-shuffled token
group, so this processor owns three things the grid-vision families do not:
its own dynamic-resolution patch budget (``image_processing_nemotron_h_omni.py``
in the released checkpoint), the antialiased bicubic resize ATen performs, and
the InternVL-style ``<img>...</img>`` placeholder that the language model
scatters projected patch embeddings onto.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from mfq.architectures.tensor_schema import RADIO_CPE_INPUT_CONTRACT
from mfq.server.vision.decode import (
    ProcessedVisionRequest,
    VisionProcessingError,
    _binary_tensors,
    _decode_data_url,
    _decode_image,
    _tensor,
)
from mfq.server.vision.resize import antialiased_bicubic_resize


class RadioVisionProcessor:
    """RADIO image path matching the released ``preprocessor_config.json``.

    Video and audio are not part of this contract: the checkpoint declares no
    video token id and the runtime reports no video capability, so the
    processor rejects those inputs instead of silently degrading them.
    """

    processor_name = "radio"
    input_contract = RADIO_CPE_INPUT_CONTRACT
    # Values below are the released checkpoint's preprocessor/config entries.
    patch_size = 16
    downsample_ratio = 0.5
    min_num_patches = 1024
    max_num_patches = 13312
    max_model_len = 16384
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    image_start_token = "<img>"
    image_token = "<image>"
    image_end_token = "</img>"
    forbidden_text_tokens = (image_start_token, image_token, image_end_token)

    def __init__(
        self,
        *,
        patch_size: int | None = None,
        downsample_ratio: float | None = None,
        min_num_patches: int | None = None,
        max_num_patches: int | None = None,
        max_model_len: int | None = None,
    ) -> None:
        if patch_size is not None:
            self.patch_size = patch_size
        if downsample_ratio is not None:
            self.downsample_ratio = downsample_ratio
        if min_num_patches is not None:
            self.min_num_patches = min_num_patches
        if max_num_patches is not None:
            self.max_num_patches = max_num_patches
        if max_model_len is not None:
            self.max_model_len = max_model_len
        if self.patch_size <= 0 or not 0.0 < self.downsample_ratio <= 1.0:
            raise VisionProcessingError("RADIO patch geometry is invalid")
        # Integer reduction of the pixel shuffle (downsample_ratio 0.5 -> 2).
        self.downsample_factor = int(round(1.0 / self.downsample_ratio))

    def tokens_available(self, image_count: int) -> int:
        """Per-image patch budget for this batch (image path, not video)."""

        if image_count <= 0:
            raise VisionProcessingError("RADIO preprocessing needs at least one image")
        budget = (self.max_model_len - 4) * (self.downsample_factor**2)
        budget = max(budget, self.min_num_patches * image_count)
        limit = self.max_num_patches if self.max_num_patches > 0 else float("inf")
        return int(max(min(budget, limit), self.min_num_patches))

    def target_patches(self, width: int, height: int, tokens_available: int) -> tuple[int, int]:
        """Choose the ``(patches_across, patches_down)`` tile for one image."""

        if width <= 0 or height <= 0:
            raise VisionProcessingError("RADIO image dimensions must be positive")
        # ``round(x + 0.5)`` follows the reference processor exactly, including
        # Python's ties-to-even result on whole-patch images.
        closest_rows = round(height / self.patch_size + 0.5)
        closest_columns = round(width / self.patch_size + 0.5)
        closest = closest_rows * closest_columns

        factor = min(math.sqrt(tokens_available / closest), 1.0)
        rows = max(1, math.floor(factor * closest_rows))
        columns = max(1, math.floor(factor * closest_columns))
        if rows * columns > tokens_available:
            raise VisionProcessingError(
                "RADIO image aspect ratio cannot fit the model's patch budget; "
                "use a less extreme aspect ratio"
            )

        if tokens_available > self.min_num_patches and rows * columns < self.min_num_patches:
            grow = math.sqrt(self.min_num_patches / (rows * columns))
            rows = math.ceil(grow * rows)
            columns = math.ceil(grow * columns)

        divisor = self.downsample_factor
        rows = self._align(rows, columns, divisor, tokens_available, by_row=True)
        columns = self._align(rows, columns, divisor, tokens_available, by_row=False)
        return columns, rows

    @staticmethod
    def _align(
        rows: int,
        columns: int,
        divisor: int,
        tokens_available: int,
        *,
        by_row: bool,
    ) -> int:
        """Round one patch dimension to the pixel-shuffle factor when it fits."""

        value = rows if by_row else columns
        remainder = value % divisor
        if not remainder:
            return value
        step = divisor - remainder
        grew = (value + step) * columns if by_row else rows * (value + step)
        if grew <= tokens_available:
            return value + step
        return max(divisor, value - remainder)

    def _prepare_image(
        self,
        image: Any,
        tokens_available: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Resize, normalize and patchify one PIL image into row-major patches."""

        image = image.convert("RGB")
        columns, rows = self.target_patches(image.width, image.height, tokens_available)
        target_height = rows * self.patch_size
        target_width = columns * self.patch_size
        pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        pixels = pixels.astype(np.float32)
        if pixels.shape[1:] != (target_height, target_width):
            pixels = antialiased_bicubic_resize(pixels, target_height, target_width)
        mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
        standard_deviation = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
        normalized = (pixels / np.float32(255.0) - mean) / standard_deviation
        return self._patchify(normalized), (1, rows, columns)

    def _patchify(self, image: np.ndarray) -> np.ndarray:
        """Flatten ``[3, H, W]`` into row-major ``[rows*columns, 3*ps*ps]`` patches.

        The element order inside a patch is channel-major (``c, row, column``),
        matching ``RadioPatchEmbeddings._image_to_patches``.
        """

        patch = self.patch_size
        channels, height, width = image.shape
        rows = height // patch
        columns = width // patch
        if rows * patch != height or columns * patch != width:
            raise VisionProcessingError("RADIO image does not divide its patch grid")
        patches = image.reshape(channels, rows, patch, columns, patch)
        patches = patches.transpose(1, 3, 0, 2, 4)
        return np.ascontiguousarray(
            patches.reshape(rows * columns, channels * patch * patch),
            dtype=np.float32,
        )

    def placeholder(self, tokens: int) -> str:
        """Template text reserved for one image's projected patch embeddings."""

        if tokens <= 0:
            raise VisionProcessingError("RADIO image produced no placeholder tokens")
        return self.image_start_token + self.image_token * tokens + self.image_end_token

    def _pack_tensors(
        self,
        patches: list[np.ndarray],
        grids: list[tuple[int, int, int]],
        *,
        use_binary_file: bool,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        if not patches or len(patches) != len(grids):
            raise VisionProcessingError("RADIO media tensor geometry is invalid")
        values = [
            ("pixel_values", np.concatenate(patches, axis=0), "float32"),
            ("vision_grid_thw", np.asarray(grids, dtype=np.int32), "int32"),
            ("vision_types", np.ones(len(grids), dtype=np.int32), "int32"),
            ("image_grid_thw", np.asarray(grids, dtype=np.int32), "int32"),
        ]
        if use_binary_file:
            tensors, path = _binary_tensors(values)
            tensors["version"] = 3
            tensors["processor"] = self.input_contract
            return tensors, (path,)
        return (
            {
                "version": 3,
                "processor": self.input_contract,
                **{name: _tensor(value, dtype) for name, value, dtype in values},
            },
            (),
        )

    def prepare_openai_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        use_binary_file: bool = False,
    ) -> ProcessedVisionRequest | None:
        prepared = copy.deepcopy(messages)
        image_count = self._validate_messages(prepared)
        if image_count == 0:
            return None
        tokens_available = self.tokens_available(image_count)
        patches: list[np.ndarray] = []
        grids: list[tuple[int, int, int]] = []
        sources = 0
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            pieces: list[str] = []
            for item in content:
                if item.get("type") == "text":
                    pieces.append(str(item.get("text", "")))
                    continue
                image_spec = item.get("image_url")
                if not isinstance(image_spec, dict) or not isinstance(
                    image_spec.get("url"), str
                ):
                    raise VisionProcessingError("image_url content is missing its URL")
                image = _decode_image(_decode_data_url(image_spec["url"], "image/"))
                pixel_values, grid = self._prepare_image(image, tokens_available)
                patches.append(pixel_values)
                grids.append(grid)
                pieces.append(self.placeholder(grid[1] * grid[2] // (self.downsample_factor**2)))
                sources += 1
            message["content"] = "".join(pieces)
        tensors, cleanup_paths = self._pack_tensors(
            patches,
            grids,
            use_binary_file=use_binary_file,
        )
        return ProcessedVisionRequest(
            messages=prepared,
            tensors=tensors,
            source_count=sources,
            frame_count=0,
            cleanup_paths=cleanup_paths,
        )

    def _validate_messages(self, messages: list[dict[str, Any]]) -> int:
        """Validate every content part and return the number of images.

        Validation runs before any image is decoded so a request that contains
        media or template text this contract cannot represent fails as a whole
        instead of after partially preparing tensors.
        """

        count = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                self._reject_template_placeholders(content)
                continue
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    raise VisionProcessingError("multimodal content item must be an object")
                item_type = item.get("type")
                if item_type == "text":
                    self._reject_template_placeholders(str(item.get("text", "")))
                    continue
                if item_type in {"video_url", "input_video", "audio_url", "input_audio"}:
                    raise VisionProcessingError(
                        f"{self.processor_name} models accept images only; "
                        f"{item_type} input is not supported"
                    )
                if item_type != "image_url":
                    raise VisionProcessingError(
                        f"{self.processor_name} does not support {item_type or 'unknown'} input"
                    )
                count += 1
        return count

    def _reject_template_placeholders(self, text: str) -> None:
        for token in self.forbidden_text_tokens:
            if token in text:
                raise VisionProcessingError(
                    f"{self.processor_name} vision placeholder tokens cannot be supplied as text"
                )


__all__ = ["RadioVisionProcessor"]
