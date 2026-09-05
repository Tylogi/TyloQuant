"""Architecture-neutral media preprocessing for native MFQ workers.

The wire format is deliberately shared while preprocessing remains owned by
the model family.  A processor may replace media blocks with prompt markers
and attach named tensors; the C++ runtime performs tokenizer-position
dependent expansion (for example DeepSeek-V4's N-layout image sentinels).
"""

from __future__ import annotations

import base64
import copy
import io
import math
import os
import platform
import secrets
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VisionProcessingError(ValueError):
    pass


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


@dataclass(frozen=True)
class _PreparedVideoFrame:
    image: Any
    source_size: tuple[int, int]


class _AVFoundationVideoDecoder:
    def __init__(self, library: str | Path) -> None:
        import ctypes

        self._ctypes = ctypes
        self._callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
        )
        self._library = ctypes.CDLL(str(library))
        function = self._library.mfq_avfoundation_sample_video
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int,
            self._callback_type,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        self._sample = function

    def decode(
        self,
        data: bytes,
        *,
        frames_per_second: float,
        maximum_frames: int,
        resize: Any,
    ) -> list[_PreparedVideoFrame]:
        import av

        descriptor, temporary_path = tempfile.mkstemp(prefix="mfq-video-", suffix=".mp4")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            with av.open(temporary_path, mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "video"]
                if not streams:
                    raise VisionProcessingError("video contains no video stream")
                stream = streams[0]
                if stream.time_base is None:
                    raise VisionProcessingError("video stream has no time base")
                time_base = stream.time_base
                codec = stream.codec_context
                color_metadata = {
                    name: getattr(codec, name)
                    for name in (
                        "color_range",
                        "colorspace",
                        "color_primaries",
                        "color_trc",
                    )
                }
                packet_pts = sorted(
                    packet.pts for packet in container.demux(stream) if packet.pts is not None
                )
            selected_pts: list[int] = []
            next_timestamp: float | None = None
            for pts in packet_pts:
                timestamp = float(pts * time_base)
                if next_timestamp is not None and timestamp + 1.0e-9 < next_timestamp:
                    continue
                selected_pts.append(pts)
                next_timestamp = timestamp + 1.0 / frames_per_second
                if len(selected_pts) >= maximum_frames:
                    break
            if not selected_pts:
                raise VisionProcessingError("video contains no decodable frames")

            ctypes = self._ctypes
            targets = (ctypes.c_int64 * len(selected_pts))(*selected_pts)
            frames: list[_PreparedVideoFrame | None] = [None] * len(selected_pts)
            callback_error: list[BaseException] = []

            def receive(
                _context: Any,
                target_index: int,
                y_plane: Any,
                y_stride: int,
                _uv_plane: Any,
                uv_stride: int,
                width: int,
                height: int,
                _presentation_seconds: float,
            ) -> int:
                try:
                    if target_index < 0 or target_index >= len(frames):
                        raise VisionProcessingError("AVFoundation returned an invalid frame index")
                    if y_stride != width or uv_stride != width:
                        raise VisionProcessingError(
                            "AVFoundation returned non-compact video planes"
                        )
                    raw = np.ctypeslib.as_array(
                        y_plane,
                        shape=(width * height * 3 // 2,),
                    ).reshape(height * 3 // 2, width)
                    frame = av.VideoFrame.from_ndarray(raw, format="nv12").reformat(
                        format="yuv420p"
                    )
                    for name, value in color_metadata.items():
                        setattr(frame, name, value)
                    image = frame.to_image().convert("RGB")
                    frames[target_index] = _PreparedVideoFrame(
                        image=resize(image),
                        source_size=image.size,
                    )
                    return 0
                except BaseException as error:
                    callback_error.append(error)
                    return 1

            callback = self._callback_type(receive)
            error_message = ctypes.create_string_buffer(1024)
            configured_parallelism = int(os.environ.get("MFQ_AVFOUNDATION_VIDEO_PARALLELISM", "16"))
            parallelism = max(1, min(32, configured_parallelism, len(selected_pts)))
            result = self._sample(
                os.fsencode(temporary_path),
                targets,
                len(selected_pts),
                time_base.numerator,
                time_base.denominator,
                parallelism,
                callback,
                None,
                error_message,
                len(error_message),
            )
            if callback_error:
                raise callback_error[0]
            if result != len(selected_pts):
                message = error_message.value.decode("utf-8", errors="replace")
                raise VisionProcessingError(
                    message or "AVFoundation did not return every selected video frame"
                )
            if any(frame is None for frame in frames):
                raise VisionProcessingError("AVFoundation omitted a selected video frame")
            return [frame for frame in frames if frame is not None]
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)


class MiniCPMO45VisionProcessor:
    """Exact CPU port of the official MiniCPM-o 4.5 media processors."""

    patch_size = 14
    scale_resolution = 448
    image_feature_size = 64
    maximum_image_slices = 9
    maximum_video_frames = 64
    video_fps = 1.0
    audio_sample_rate = 16_000
    audio_chunk_samples = 30 * audio_sample_rate
    minimum_audio_samples = audio_sample_rate // 10
    maximum_audio_samples = 30 * 60 * audio_sample_rate

    def __init__(self, avfoundation_library: str | Path | None = None) -> None:
        library = avfoundation_library or os.environ.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
        self._avfoundation_decoder: _AVFoundationVideoDecoder | None = None
        self._mel_extractor: Any | None = None
        if platform.system() == "Darwin" and library and Path(library).is_file():
            try:
                self._avfoundation_decoder = _AVFoundationVideoDecoder(library)
            except (OSError, AttributeError):
                self._avfoundation_decoder = None

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

    @staticmethod
    def _decode_audio_data(value: Any) -> bytes:
        if not isinstance(value, dict) or not isinstance(value.get("data"), str):
            raise VisionProcessingError("input_audio content is missing its base64 data")
        try:
            data = base64.b64decode(value["data"], validate=True)
        except ValueError as error:
            raise VisionProcessingError("audio input contains invalid base64") from error
        if not data:
            raise VisionProcessingError("audio input is empty")
        return data

    @classmethod
    def _decode_audio(cls, data: bytes) -> np.ndarray:
        try:
            import av
        except ImportError as error:
            raise VisionProcessingError(
                "audio input requires the optional PyAV dependency"
            ) from error

        chunks: list[np.ndarray] = []
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "audio"]
                if not streams:
                    raise VisionProcessingError("audio contains no audio stream")
                resampler = av.AudioResampler(
                    format="fltp",
                    layout="mono",
                    rate=cls.audio_sample_rate,
                )

                def append_frames(frames: Any) -> None:
                    if frames is None:
                        return
                    if not isinstance(frames, list):
                        frames = [frames]
                    for frame in frames:
                        if frame is not None:
                            chunks.append(
                                np.ascontiguousarray(
                                    frame.to_ndarray().reshape(-1),
                                    dtype=np.float32,
                                )
                            )

                for frame in container.decode(streams[0]):
                    append_frames(resampler.resample(frame))
                append_frames(resampler.resample(None))
        except VisionProcessingError:
            raise
        except Exception as error:
            raise VisionProcessingError(f"unable to decode audio: {error}") from error
        if not chunks:
            raise VisionProcessingError("audio contains no decodable samples")
        waveform = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        if waveform.size > cls.maximum_audio_samples:
            raise VisionProcessingError("audio input exceeds the 30 minute limit")
        if not np.isfinite(waveform).all():
            raise VisionProcessingError("audio input contains a non-finite sample")
        return np.clip(waveform, -1.0, 1.0)

    @staticmethod
    def _audio_placeholder(frame_count: int) -> str:
        after_convolution = (frame_count - 1) // 2 + 1
        pooled = (after_convolution - 5) // 5 + 1
        if frame_count <= 0 or pooled <= 0:
            raise VisionProcessingError("audio input is too short")
        return f"<|audio_start|>{'<unk>' * pooled}<|audio_end|>"

    def _prepare_audio(self, data: bytes) -> list[np.ndarray]:
        from mfq.runtime.minicpmo45_realtime import MiniCPMOMel

        waveform = self._decode_audio(data)
        if waveform.size < self.minimum_audio_samples:
            waveform = np.pad(
                waveform,
                (0, self.minimum_audio_samples - waveform.size),
            )
        if self._mel_extractor is None:
            self._mel_extractor = MiniCPMOMel()
        result: list[np.ndarray] = []
        for start in range(0, waveform.size, self.audio_chunk_samples):
            chunk = waveform[start : start + self.audio_chunk_samples]
            if chunk.size < self.minimum_audio_samples:
                chunk = np.pad(chunk, (0, self.minimum_audio_samples - chunk.size))
            features = self._mel_extractor.extract(chunk, fixed_floor=False)
            if features.shape[0] != 80 or not 9 <= features.shape[1] <= 3000:
                raise VisionProcessingError("processed audio tensor geometry is invalid")
            result.append(np.ascontiguousarray(features, dtype=np.float32))
        return result

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
                    average_rate = float(stream.average_rate) if stream.average_rate else 30.0
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

    @classmethod
    def _resize_video_frame(cls, image: Any) -> Any:
        from PIL import Image

        return image.resize(
            cls._best_resize(image.size, allow_upscale=True),
            resample=Image.Resampling.BICUBIC,
        )

    def _decode_video_for_request(self, data: bytes) -> list[_PreparedVideoFrame]:
        decoder = self._avfoundation_decoder
        if decoder is not None:
            try:
                return decoder.decode(
                    data,
                    frames_per_second=self.video_fps,
                    maximum_frames=self.maximum_video_frames,
                    resize=self._resize_video_frame,
                )
            except (OSError, RuntimeError, VisionProcessingError, ValueError):
                pass
        return [
            _PreparedVideoFrame(
                image=self._resize_video_frame(image),
                source_size=image.size,
            )
            for image in self._decode_video(data)
        ]

    @staticmethod
    def _packed_tensor(value: np.ndarray, dtype: str) -> np.ndarray:
        little_endian = {
            "float32": "<f4",
            "int32": "<i4",
            "int64": "<i8",
            "uint8": "u1",
        }[dtype]
        return np.ascontiguousarray(value, dtype=little_endian)

    @classmethod
    def _tensor(cls, value: np.ndarray, dtype: str) -> dict[str, Any]:
        packed = cls._packed_tensor(value, dtype)
        return {
            "dtype": dtype,
            "shape": list(packed.shape),
            "data_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
        }

    @classmethod
    def _binary_tensors(
        cls,
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
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(magic)
                stream.write(token)
                stream.write(bytes(64 - len(magic) - len(token)))
                for name, value, dtype in values:
                    packed = cls._packed_tensor(value, dtype)
                    offset = stream.tell()
                    padding = (-offset) % 64
                    if padding:
                        stream.write(bytes(padding))
                        offset += padding
                    raw = packed.tobytes()
                    stream.write(raw)
                    tensors[name] = {
                        "dtype": dtype,
                        "shape": list(packed.shape),
                        "data_offset": offset,
                        "data_length": len(raw),
                    }
                tensors["binary_file"]["size"] = stream.tell()
            return tensors, path
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            path.unlink(missing_ok=True)
            raise

    @classmethod
    def _pack_tensors(
        cls,
        patches: list[np.ndarray],
        target_sizes: list[tuple[int, int]],
        audio_features: list[np.ndarray],
        *,
        use_binary_file: bool = False,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        values: list[tuple[str, np.ndarray, str]] = []
        if patches or target_sizes:
            if not patches or len(patches) != len(target_sizes):
                raise VisionProcessingError("processed image tensor geometry is invalid")
            sequences = [patch.reshape(3 * cls.patch_size, -1).T for patch in patches]
            maximum_length = max(sequence.shape[0] for sequence in sequences)
            padded = np.zeros(
                (len(sequences), maximum_length, 3 * cls.patch_size),
                dtype=np.float32,
            )
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
            values.extend(
                [
                    ("pixel_values", pixels, "float32"),
                    ("patch_mask", mask, "uint8"),
                    ("target_sizes", sizes, "int32"),
                ]
            )
        if audio_features:
            maximum_frames = max(features.shape[1] for features in audio_features)
            padded_audio = np.zeros(
                (len(audio_features), 80, maximum_frames),
                dtype=np.float32,
            )
            lengths = np.empty(len(audio_features), dtype=np.int64)
            for index, features in enumerate(audio_features):
                if features.ndim != 2 or features.shape[0] != 80:
                    raise VisionProcessingError("processed audio tensor geometry is invalid")
                padded_audio[index, :, : features.shape[1]] = features
                lengths[index] = features.shape[1]
            values.extend(
                [
                    ("audio_features", padded_audio, "float32"),
                    ("audio_lengths", lengths, "int64"),
                ]
            )
        if not values:
            raise VisionProcessingError("multimodal request contains no processed media")
        if use_binary_file:
            tensors, path = cls._binary_tensors(values)
            return tensors, (path,)
        return (
            {
                "version": 1,
                **{name: cls._tensor(value, dtype) for name, value, dtype in values},
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
        patches: list[np.ndarray] = []
        target_sizes: list[tuple[int, int]] = []
        audio_features: list[np.ndarray] = []
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
                    video_frames = self._decode_video_for_request(
                        self._decode_data_url(video_spec["url"], "video/")
                    )
                    images = [frame.image for frame in video_frames]
                    maximum_slices = 1
                    source_count += 1
                    frame_count += len(images)
                elif item_type == "input_audio":
                    features_list = self._prepare_audio(
                        self._decode_audio_data(item.get("input_audio"))
                    )
                    source_count += 1
                    found_media = True
                    for features in features_list:
                        pieces.append(self._audio_placeholder(features.shape[1]))
                        audio_features.append(features)
                    continue
                else:
                    raise VisionProcessingError(f"unsupported multimodal content type: {item_type}")
                found_media = True
                for media_index, image in enumerate(images):
                    source_size = (
                        video_frames[media_index].source_size
                        if item_type == "video_url"
                        else image.size
                    )
                    pieces.append(self._placeholder(source_size, image_index, maximum_slices))
                    sliced = (
                        [image]
                        if item_type == "video_url"
                        else self._slice_image(image, maximum_slices)[0]
                    )
                    for image_slice in sliced:
                        packed, target = self._reshape_by_patch(image_slice)
                        patches.append(packed)
                        target_sizes.append(target)
                    image_index += 1
            message["content"] = "\n".join(piece for piece in pieces if piece)
        if not found_media:
            return None
        tensors, cleanup_paths = self._pack_tensors(
            patches,
            target_sizes,
            audio_features,
            use_binary_file=use_binary_file,
        )
        return ProcessedVisionRequest(
            messages=prepared,
            tensors=tensors,
            source_count=source_count,
            frame_count=frame_count,
            cleanup_paths=cleanup_paths,
        )


class DeepseekV4VisionProcessor:
    """Exact image preprocessing contract released with DSV4 Vision.

    The processor intentionally leaves ``<｜deepseek_image｜>`` in the prompt.
    The final image block depends on the marker's *token* position, so the
    native worker expands it only after applying the chat template and
    tokenizing the prompt.
    """

    image_placeholder = "<｜deepseek_image｜>"
    patch_size = 14
    downsample_ratio = 3
    maximum_image_tokens = 384
    minimum_pixels = 147_456
    maximum_width_height_ratio = 8

    @classmethod
    def _grid_tokens(
        cls,
        best_height: int,
        best_width: int,
    ) -> tuple[int, int, int]:
        n_llm_h = math.ceil((best_height // cls.patch_size) / cls.downsample_ratio)
        n_llm_w = math.ceil((best_width // cls.patch_size) / cls.downsample_ratio)
        count = n_llm_h * (n_llm_w + 1) + 2
        if n_llm_h % 2 == 1:
            count += n_llm_w + 1
        count += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
        return n_llm_h, n_llm_w, count

    @classmethod
    def _solve_resize_ratio(
        cls,
        height: int,
        width: int,
        budget: int,
    ) -> tuple[int, int, int, int, int]:
        ratio = height / width
        max_w_float = math.sqrt((budget - 2) / ratio + 0.25) - 0.5
        max_h_float = max_w_float * ratio
        unit = cls.patch_size * cls.downsample_ratio
        if max_w_float < 1.0:
            max_w = 1
            max_h = (budget - 2) // (max_w + 1)
            if max_h % 2 == 1:
                max_h -= 1
            best_width = max_w * unit
            best_height = max_h * unit
        elif max_h_float < 2.0:
            max_h = 2
            max_w = (budget - 2) // max_h - 1
            if max_w <= 1:
                raise VisionProcessingError("image token budget is too small")
            best_width = max_w * unit
            best_height = max_h * unit
        else:
            max_w = math.floor(max_w_float)
            max_h = math.floor(max_h_float)
            if max_h % 2 == 1:
                max_h -= 1
            beta = min(max_w * unit / width, max_h * unit / height)
            best_width = math.floor(width * beta / cls.patch_size) * cls.patch_size
            best_height = math.floor(height * beta / cls.patch_size) * cls.patch_size
        n_llm_h, n_llm_w, count = cls._grid_tokens(best_height, best_width)
        return n_llm_h, n_llm_w, best_height, best_width, count

    @classmethod
    def _safe_resize(
        cls,
        height: int,
        width: int,
        best_height: int,
        best_width: int,
    ) -> tuple[int, int, int, int]:
        maximum = cls.maximum_image_tokens - 3
        n_llm_h, n_llm_w, count = cls._grid_tokens(best_height, best_width)
        budget = maximum
        while count > maximum:
            n_llm_h, n_llm_w, best_height, best_width, count = cls._solve_resize_ratio(
                height, width, budget
            )
            budget -= 1
            if budget <= 4:
                raise VisionProcessingError("unable to fit image in the token budget")
        return n_llm_h, n_llm_w, best_height, best_width

    @classmethod
    def _prepare_image(
        cls,
        image: Any,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        from PIL import ImageOps

        image = image.convert("RGB")
        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise VisionProcessingError("image dimensions must be positive")
        width, height = source_width, source_height
        if width > height * cls.maximum_width_height_ratio:
            width = height * cls.maximum_width_height_ratio
        if width * height < cls.minimum_pixels:
            ratio = math.sqrt(cls.minimum_pixels / (width * height))
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / cls.patch_size) * cls.patch_size
        best_height = math.ceil(height / cls.patch_size) * cls.patch_size
        n_llm_h, n_llm_w, best_height, best_width = cls._safe_resize(
            height,
            width,
            best_height,
            best_width,
        )
        n_vit_h = best_height // cls.patch_size
        n_vit_w = best_width // cls.patch_size
        if source_width >= cls.maximum_width_height_ratio * source_height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image,
                (best_width, best_height),
                color=(127, 127, 127),
            )
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        pixels = (pixels / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
        patches = (
            pixels.reshape(
                3,
                n_vit_h,
                cls.patch_size,
                n_vit_w,
                cls.patch_size,
            )
            .transpose(1, 3, 0, 2, 4)
            .reshape(-1, 3, cls.patch_size, cls.patch_size)
        )
        return (
            np.ascontiguousarray(patches, dtype=np.float32),
            (n_vit_h, n_vit_w, n_llm_h, n_llm_w),
        )

    @classmethod
    def _pack_tensors(
        cls,
        patches: list[np.ndarray],
        grids: list[tuple[int, int, int, int]],
        *,
        use_binary_file: bool,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        if not patches or len(patches) != len(grids):
            raise VisionProcessingError("processed image tensor geometry is invalid")
        maximum_patches = max(value.shape[0] for value in patches)
        pixels = np.zeros(
            (
                len(patches),
                maximum_patches,
                3 * cls.patch_size * cls.patch_size,
            ),
            dtype=np.float32,
        )
        mask = np.zeros((len(patches), maximum_patches), dtype=np.uint8)
        for index, value in enumerate(patches):
            pixels[index, : value.shape[0]] = value.reshape(value.shape[0], -1)
            mask[index, : value.shape[0]] = 1
        values = [
            ("pixel_values", pixels, "float32"),
            ("patch_mask", mask, "uint8"),
            ("vision_grid", np.asarray(grids, dtype=np.int32), "int32"),
        ]
        if use_binary_file:
            tensors, path = MiniCPMO45VisionProcessor._binary_tensors(values)
            tensors["version"] = 2
            tensors["processor"] = "deepseek_v4"
            return tensors, (path,)
        return (
            {
                "version": 2,
                "processor": "deepseek_v4",
                **{
                    name: MiniCPMO45VisionProcessor._tensor(value, dtype)
                    for name, value, dtype in values
                },
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
        patches: list[np.ndarray] = []
        grids: list[tuple[int, int, int, int]] = []
        sources = 0
        found_media = False
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                if isinstance(content, str) and self.image_placeholder in content:
                    raise VisionProcessingError(
                        "image placeholder tokens cannot be supplied as text"
                    )
                continue
            pieces: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise VisionProcessingError("multimodal content item must be an object")
                item_type = item.get("type")
                if item_type == "text":
                    value = str(item.get("text", ""))
                    if self.image_placeholder in value:
                        raise VisionProcessingError(
                            "image placeholder tokens cannot be supplied as text"
                        )
                    pieces.append(value)
                    continue
                if item_type != "image_url":
                    raise VisionProcessingError(
                        f"DeepSeek-V4 Vision does not support {item_type or 'unknown'} input"
                    )
                image_spec = item.get("image_url")
                if not isinstance(image_spec, dict) or not isinstance(image_spec.get("url"), str):
                    raise VisionProcessingError("image_url content is missing its URL")
                image = MiniCPMO45VisionProcessor._decode_image(
                    MiniCPMO45VisionProcessor._decode_data_url(
                        image_spec["url"],
                        "image/",
                    )
                )
                value, grid = self._prepare_image(image)
                patches.append(value)
                grids.append(grid)
                pieces.append(self.image_placeholder)
                sources += 1
                found_media = True
            message["content"] = "\n\n".join(piece for piece in pieces if piece)
        if not found_media:
            return None
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


def multimodal_processor_for_architecture(
    model_type: str,
    *,
    avfoundation_library: str | Path | None = None,
) -> MultimodalProcessor | None:
    """Create the media processor registered for a runtime architecture."""

    identity = "_".join(
        part for part in model_type.strip().lower().replace("-", "_").split("_") if part
    )
    if identity == "minicpmo":
        return MiniCPMO45VisionProcessor(avfoundation_library=avfoundation_library)
    if identity in {
        "deepseek_v4",
        "deepseekv4",
        "deepseek_v4_vision",
    }:
        return DeepseekV4VisionProcessor()
    return None


__all__ = [
    "DeepseekV4VisionProcessor",
    "MiniCPMO45VisionProcessor",
    "MultimodalProcessor",
    "ProcessedVisionRequest",
    "VisionProcessingError",
    "multimodal_processor_for_architecture",
]
