"""Shared MiniCPM-o image and video preprocessing for native MFQ workers."""

from __future__ import annotations

import base64
import copy
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class VisionProcessingError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessedVisionRequest:
    messages: list[dict[str, Any]]
    tensors: dict[str, Any]
    source_count: int
    frame_count: int


class MiniCPMO45VisionProcessor:
    """Exact CPU port of the official MiniCPM-o 4.5 image processor."""

    patch_size = 14
    scale_resolution = 448
    image_feature_size = 64
    maximum_image_slices = 9
    maximum_video_frames = 64
    video_fps = 1.0

    @staticmethod
    def _ensure_divide(length: int, divisor: int) -> int:
        return max(round(length / divisor) * divisor, divisor)

    @classmethod
    def _best_resize(
        cls,
        size: tuple[int, int],
        *,
        allow_upscale: bool,
    ) -> tuple[int, int]:
        width, height = size
        if width <= 0 or height <= 0:
            raise VisionProcessingError("image dimensions must be positive")
        if width * height > cls.scale_resolution**2 or allow_upscale:
            ratio = width / height
            height = int(cls.scale_resolution / math.sqrt(ratio))
            width = int(height * ratio)
        return (
            cls._ensure_divide(width, cls.patch_size),
            cls._ensure_divide(height, cls.patch_size),
        )

    @classmethod
    def _sliced_grid(
        cls,
        size: tuple[int, int],
        maximum_slices: int,
    ) -> tuple[int, int] | None:
        width, height = size
        ratio = width * height / cls.scale_resolution**2
        multiple = min(math.ceil(ratio), maximum_slices)
        if multiple <= 1:
            return None
        candidates: list[tuple[int, int]] = []
        for count in (multiple - 1, multiple, multiple + 1):
            if count == 1 or count > maximum_slices:
                continue
            for columns in range(1, count + 1):
                if count % columns == 0:
                    candidates.append((columns, count // columns))
        if not candidates:
            return None
        log_ratio = math.log(width / height)
        return min(candidates, key=lambda grid: abs(log_ratio - math.log(grid[0] / grid[1])))

    @classmethod
    def _refine_size(
        cls,
        size: tuple[int, int],
        grid: tuple[int, int],
    ) -> tuple[int, int]:
        width, height = size
        columns, rows = grid
        refined_width = cls._ensure_divide(width, columns)
        refined_height = cls._ensure_divide(height, rows)
        tile = cls._best_resize(
            (refined_width / columns, refined_height / rows),
            allow_upscale=True,
        )
        return tile[0] * columns, tile[1] * rows

    @classmethod
    def _slice_image(cls, image: Any, maximum_slices: int) -> tuple[list[Any], Any]:
        from PIL import Image

        image = image.convert("RGB")
        grid = cls._sliced_grid(image.size, maximum_slices)
        overview = image.resize(
            cls._best_resize(image.size, allow_upscale=grid is None),
            resample=Image.Resampling.BICUBIC,
        )
        slices = [overview]
        if grid is not None:
            refined = image.resize(
                cls._refine_size(image.size, grid),
                resample=Image.Resampling.BICUBIC,
            )
            columns, rows = grid
            tile_width = refined.width // columns
            tile_height = refined.height // rows
            for row in range(rows):
                for column in range(columns):
                    slices.append(
                        refined.crop(
                            (
                                column * tile_width,
                                row * tile_height,
                                (column + 1) * tile_width,
                                (row + 1) * tile_height,
                            )
                        )
                    )
        return slices, grid

    @classmethod
    def _reshape_by_patch(cls, image: Any) -> tuple[np.ndarray, tuple[int, int]]:
        value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        value = (value - np.float32(0.5)) / np.float32(0.5)
        value = np.transpose(value, (2, 0, 1))
        channels, height, width = value.shape
        patch = cls.patch_size
        if height % patch or width % patch:
            raise VisionProcessingError("processed image is not divisible by the patch size")
        rows = height // patch
        columns = width // patch
        packed = (
            value.reshape(channels, rows, patch, columns, patch)
            .transpose(0, 2, 1, 3, 4)
            .reshape(channels, patch, rows * columns * patch)
        )
        return np.ascontiguousarray(packed, dtype=np.float32), (rows, columns)

    @classmethod
    def _placeholder(
        cls,
        size: tuple[int, int],
        image_index: int,
        maximum_slices: int,
    ) -> str:
        unknowns = "<unk>" * cls.image_feature_size
        result = f"<image_id>{image_index}</image_id><image>{unknowns}</image>"
        grid = cls._sliced_grid(size, maximum_slices)
        if grid is None:
            return result
        columns, rows = grid
        slice_placeholder = f"<slice>{unknowns}</slice>"
        return result + "\n".join(slice_placeholder * columns for _ in range(rows))

    @staticmethod
    def _decode_data_url(url: str, expected_prefix: str) -> bytes:
        if not url.startswith("data:") or ";base64," not in url:
            raise VisionProcessingError("native vision input requires a base64 data URL")
        header, encoded = url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].lower()
        if not mime_type.startswith(expected_prefix):
            raise VisionProcessingError(f"expected {expected_prefix} media, got {mime_type}")
        try:
            return base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise VisionProcessingError("media data URL contains invalid base64") from error

    @staticmethod
    def _decode_image(data: bytes) -> Any:
        from PIL import Image

        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            return image.convert("RGB")
        except Exception as error:
            raise VisionProcessingError(f"unable to decode image: {error}") from error

    @classmethod
    def _decode_video(cls, data: bytes) -> list[Any]:
        try:
            import av
        except ImportError as error:
            raise VisionProcessingError(
                "video input requires the optional PyAV dependency"
            ) from error

        def open_stream() -> tuple[Any, Any]:
            container = av.open(io.BytesIO(data), mode="r")
            streams = [stream for stream in container.streams if stream.type == "video"]
            if not streams:
                container.close()
                raise VisionProcessingError("video contains no video stream")
            stream = streams[0]
            # PyAV defaults to slice-only decoding. Frame threading makes
            # inter-frame codecs use the available CPU cores without changing
            # decoded pixels or presentation timestamps.
            stream.thread_type = "FRAME"
            return container, stream

        def timestamp_of(frame: Any, index: int, average_rate: float) -> float:
            if frame.time is not None:
                return float(frame.time)
            if frame.pts is not None and frame.time_base is not None:
                return float(frame.pts * frame.time_base)
            return index / max(average_rate, 1.0)

        def decode_sequentially() -> list[Any]:
            selected: list[Any] = []
            next_timestamp = 0.0
            container, stream = open_stream()
            try:
                average_rate = float(stream.average_rate) if stream.average_rate else 30.0
                for index, frame in enumerate(container.decode(stream)):
                    timestamp = timestamp_of(frame, index, average_rate)
                    if selected and timestamp + 1.0e-9 < next_timestamp:
                        continue
                    selected.append(frame.to_image().convert("RGB"))
                    next_timestamp = timestamp + 1.0 / cls.video_fps
                    if len(selected) >= cls.maximum_video_frames:
                        break
            finally:
                container.close()
            return selected

        frames: list[Any] = []
        try:
            try:
                # Indexed containers can jump to the keyframe preceding each
                # requested timestamp. This avoids decoding long spans that
                # will never be sampled. If seeking is unavailable, the exact
                # sequential path remains the compatibility fallback.
                container, stream = open_stream()
                try:
                    if stream.time_base is None:
                        raise RuntimeError("video stream has no seek time base")
                    time_base = float(stream.time_base)
                    average_rate = (
                        float(stream.average_rate) if stream.average_rate else 30.0
                    )
                    next_timestamp = 0.0
                    while len(frames) < cls.maximum_video_frames:
                        container.seek(
                            max(0, int(next_timestamp / time_base)),
                            stream=stream,
                            backward=True,
                            any_frame=False,
                        )
                        selected = None
                        for index, frame in enumerate(container.decode(stream)):
                            timestamp = timestamp_of(frame, index, average_rate)
                            if timestamp + 1.0e-9 >= next_timestamp:
                                selected = (frame, timestamp)
                                break
                        if selected is None:
                            break
                        frame, timestamp = selected
                        frames.append(frame.to_image().convert("RGB"))
                        next_timestamp = timestamp + 1.0 / cls.video_fps
                finally:
                    container.close()
            except VisionProcessingError:
                raise
            except Exception:
                frames = decode_sequentially()
            if not frames:
                frames = decode_sequentially()
        except VisionProcessingError:
            raise
        except Exception as error:
            raise VisionProcessingError(f"unable to decode video: {error}") from error
        if not frames:
            raise VisionProcessingError("video contains no decodable frames")
        return frames

    @staticmethod
    def _tensor(value: np.ndarray, dtype: str) -> dict[str, Any]:
        little_endian = {
            "float32": "<f4",
            "int32": "<i4",
            "uint8": "u1",
        }[dtype]
        packed = np.ascontiguousarray(value, dtype=little_endian)
        return {
            "dtype": dtype,
            "shape": list(packed.shape),
            "data_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
        }

    @classmethod
    def _pack_tensors(
        cls,
        patches: list[np.ndarray],
        target_sizes: list[tuple[int, int]],
    ) -> dict[str, Any]:
        if not patches or len(patches) != len(target_sizes):
            raise VisionProcessingError("vision request contains no processed image patches")
        sequences = [patch.reshape(3 * cls.patch_size, -1).T for patch in patches]
        maximum_length = max(sequence.shape[0] for sequence in sequences)
        padded = np.zeros((len(sequences), maximum_length, 3 * cls.patch_size), dtype=np.float32)
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.shape[0]] = sequence
        pixels = padded.transpose(0, 2, 1).reshape(
            len(sequences), 3, cls.patch_size, maximum_length
        )
        sizes = np.asarray(target_sizes, dtype=np.int32)
        maximum_patches = int(np.max(sizes[:, 0] * sizes[:, 1]))
        mask = np.zeros((len(sequences), maximum_patches), dtype=np.uint8)
        for index, (rows, columns) in enumerate(target_sizes):
            mask[index, : rows * columns] = 1
        return {
            "version": 1,
            "pixel_values": cls._tensor(pixels, "float32"),
            "patch_mask": cls._tensor(mask, "uint8"),
            "target_sizes": cls._tensor(sizes, "int32"),
        }

    def prepare_openai_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> ProcessedVisionRequest | None:
        prepared = copy.deepcopy(messages)
        patches: list[np.ndarray] = []
        target_sizes: list[tuple[int, int]] = []
        source_count = 0
        frame_count = 0
        image_index = 0
        found_media = False
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            pieces: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise VisionProcessingError("multimodal content item must be an object")
                item_type = item.get("type")
                if item_type == "text":
                    pieces.append(str(item.get("text", "")))
                    continue
                if item_type == "image_url":
                    image_spec = item.get("image_url")
                    if not isinstance(image_spec, dict) or not isinstance(
                        image_spec.get("url"), str
                    ):
                        raise VisionProcessingError("image_url content is missing its URL")
                    images = [
                        self._decode_image(self._decode_data_url(image_spec["url"], "image/"))
                    ]
                    maximum_slices = self.maximum_image_slices
                    source_count += 1
                elif item_type == "video_url":
                    video_spec = item.get("video_url")
                    if not isinstance(video_spec, dict) or not isinstance(
                        video_spec.get("url"), str
                    ):
                        raise VisionProcessingError("video_url content is missing its URL")
                    images = self._decode_video(self._decode_data_url(video_spec["url"], "video/"))
                    maximum_slices = 1
                    source_count += 1
                    frame_count += len(images)
                elif item_type == "input_audio":
                    raise VisionProcessingError(
                        "recorded audio preprocessing is not part of the native vision request"
                    )
                else:
                    raise VisionProcessingError(f"unsupported multimodal content type: {item_type}")
                found_media = True
                for image in images:
                    pieces.append(self._placeholder(image.size, image_index, maximum_slices))
                    sliced, _ = self._slice_image(image, maximum_slices)
                    for image_slice in sliced:
                        packed, target = self._reshape_by_patch(image_slice)
                        patches.append(packed)
                        target_sizes.append(target)
                    image_index += 1
            message["content"] = "\n".join(piece for piece in pieces if piece)
        if not found_media:
            return None
        return ProcessedVisionRequest(
            messages=prepared,
            tensors=self._pack_tensors(patches, target_sizes),
            source_count=source_count,
            frame_count=frame_count,
        )
