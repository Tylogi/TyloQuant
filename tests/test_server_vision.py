from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import stat
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from mfq.server.backend import OpenAIChatBackend
from mfq.server.models import SamplingParams
from mfq.server.vision import (
    DeepseekV41VisionProcessor,
    DeepseekV4VisionProcessor,
    Glm5NextVisionProcessor,
    MiniCPMO45VisionProcessor,
    Qwen4ExpVisionProcessor,
    Qwen35VisionProcessor,
    _DecodedVideo,
    _PreparedVideoFrame,
    multimodal_processor_for_architecture,
)


def _data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_tensor(tensor: dict[str, object]) -> np.ndarray:
    dtypes = {
        "float32": "<f4",
        "int32": "<i4",
        "int64": "<i8",
        "uint8": "u1",
    }
    raw = base64.b64decode(str(tensor["data_base64"]), validate=True)
    return np.frombuffer(raw, dtype=dtypes[str(tensor["dtype"])]).reshape(tuple(tensor["shape"]))


def _decode_binary_tensor(tensors: dict[str, object], name: str) -> np.ndarray:
    dtypes = {
        "float32": "<f4",
        "int32": "<i4",
        "int64": "<i8",
        "uint8": "u1",
    }
    file_spec = tensors["binary_file"]
    tensor = tensors[name]
    assert isinstance(file_spec, dict)
    assert isinstance(tensor, dict)
    with open(str(file_spec["path"]), "rb") as stream:
        stream.seek(int(tensor["data_offset"]))
        raw = stream.read(int(tensor["data_length"]))
    return np.frombuffer(raw, dtype=dtypes[str(tensor["dtype"])]).reshape(tuple(tensor["shape"]))


def test_patch_layout_matches_official_torch_unfold() -> None:
    height, width = 42, 70
    pixels = np.arange(height * width * 3, dtype=np.uint32)
    pixels = (pixels % 256).astype(np.uint8).reshape(height, width, 3)
    image = Image.fromarray(pixels, mode="RGB")

    actual, target = MiniCPMO45VisionProcessor._reshape_by_patch(image)
    normalized = torch.from_numpy(
        (pixels.astype(np.float32) / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
    ).permute(2, 0, 1)
    reference = torch.nn.functional.unfold(normalized, (14, 14), stride=(14, 14))
    reference = reference.reshape(3, 14, 14, -1).permute(0, 1, 3, 2).reshape(3, 14, -1).numpy()

    assert target == (3, 5)
    np.testing.assert_array_equal(actual, reference)


def test_deepseek_v4_processor_matches_official_patch_layout() -> None:
    class TinyProcessor(DeepseekV4VisionProcessor):
        minimum_pixels = 0

    height, width = 42, 70
    pixels = np.arange(height * width * 3, dtype=np.uint32)
    pixels = (pixels % 256).astype(np.uint8).reshape(height, width, 3)
    image = Image.fromarray(pixels, mode="RGB")

    actual, grid = TinyProcessor._prepare_image(image)
    normalized = torch.from_numpy(pixels.astype(np.float32)).permute(2, 0, 1)
    normalized = (normalized / 255 - 0.5) / 0.5
    reference = (
        normalized.reshape(3, 3, 14, 5, 14).permute(1, 3, 0, 2, 4).reshape(15, 3, 14, 14).numpy()
    )

    assert grid == (3, 5, 1, 2)
    np.testing.assert_array_equal(actual, reference)


def test_deepseek_v4_request_defers_position_dependent_image_block() -> None:
    class TinyProcessor(DeepseekV4VisionProcessor):
        minimum_pixels = 0

    image = Image.new("RGB", (70, 42), (20, 40, 60))
    result = TinyProcessor().prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                ],
            }
        ]
    )

    assert result is not None
    assert result.messages[0]["content"].endswith("<｜deepseek_image｜>")
    assert result.tensors["version"] == 2
    assert result.tensors["processor"] == "deepseek_v4"
    assert _decode_tensor(result.tensors["pixel_values"]).shape == (1, 15, 588)
    np.testing.assert_array_equal(
        _decode_tensor(result.tensors["patch_mask"]).sum(axis=1),
        [15],
    )
    np.testing.assert_array_equal(
        _decode_tensor(result.tensors["vision_grid"]),
        [[3, 5, 1, 2]],
    )


def test_deepseek_v41_processor_uses_released_row_major_contract() -> None:
    class TinyProcessor(DeepseekV41VisionProcessor):
        minimum_pixels = 0

    image = Image.new("RGB", (70, 42), (20, 40, 60))
    result = TinyProcessor().prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                ],
            }
        ]
    )

    assert result is not None
    assert result.tensors["version"] == 2
    assert result.tensors["processor"] == "deepseek_v41"
    assert _decode_tensor(result.tensors["pixel_values"]).shape == (1, 15, 588)
    np.testing.assert_array_equal(
        _decode_tensor(result.tensors["vision_grid"]),
        [[3, 5, 1, 2]],
    )
    # START + two image cells + NEWLINE + END.
    assert TinyProcessor._grid_tokens(42, 70) == (1, 2, 5)


def test_qwen4_exp_image_processor_matches_official_block_major_layout() -> None:
    class TinyProcessor(Qwen4ExpVisionProcessor):
        minimum_pixels = 0

    height, width = 64, 96
    pixels = np.arange(height * width * 3, dtype=np.uint32)
    pixels = (pixels % 256).astype(np.uint8).reshape(height, width, 3)
    image = Image.fromarray(pixels, mode="RGB")
    result = TinyProcessor().prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    {"type": "text", "text": "Describe it."},
                ],
            }
        ]
    )

    assert result is not None
    actual = _decode_tensor(result.tensors["pixel_values"])
    normalized = pixels.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    reference = np.stack((normalized, normalized), axis=0)[None]
    reference = reference.reshape(1, 1, 2, 3, 2, 2, 16, 3, 2, 16)
    reference = reference.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    reference = reference.reshape(24, 3 * 2 * 16 * 16)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(
        _decode_tensor(result.tensors["image_grid_thw"]),
        [[1, 4, 6]],
    )
    assert result.messages[0]["content"].count("<|image_pad|>") == 6
    assert result.messages[0]["content"].endswith("<|vision_end|>Describe it.")


def test_glm5_next_image_processor_matches_official_padded_patch_layout() -> None:
    class TinyProcessor(Glm5NextVisionProcessor):
        minimum_tokens = 0

    height, width = 56, 84
    pixels = np.arange(height * width * 3, dtype=np.uint32)
    pixels = (pixels % 256).astype(np.uint8).reshape(height, width, 3)
    image = Image.fromarray(pixels, mode="RGB")
    result = TinyProcessor().prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look: "},
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                ],
            }
        ]
    )

    assert result is not None
    actual = _decode_tensor(result.tensors["pixel_values"])
    channel_first = pixels.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(TinyProcessor.image_mean, dtype=np.float32)[:, None, None]
    standard_deviation = np.asarray(
        TinyProcessor.image_std,
        dtype=np.float32,
    )[:, None, None]
    normalized = (channel_first - mean) / standard_deviation
    reference = normalized.reshape(3, 2, 2, 14, 3, 2, 14)
    reference = reference.transpose(1, 4, 2, 5, 0, 3, 6)
    reference = np.broadcast_to(
        reference[:, :, :, :, :, None, :, :],
        (*reference.shape[:5], 2, *reference.shape[5:]),
    ).reshape(24, 3 * 2 * 14 * 14)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(
        _decode_tensor(result.tensors["image_grid_thw"]),
        [[1, 4, 6]],
    )
    assert result.messages[0]["content"].count("<|image|>") == 6
    assert result.messages[0]["content"].startswith("Look: <|begin_of_image|>")


def _decoded_video(
    frames: list[np.ndarray],
    *,
    frame_indices: tuple[int, ...],
    frames_per_second: float,
) -> _DecodedVideo:
    prepared = tuple(
        _PreparedVideoFrame(
            image=Image.fromarray(frame, mode="RGB"),
            source_size=(frame.shape[1], frame.shape[0]),
            presentation_seconds=source_index / frames_per_second,
        )
        for frame, source_index in zip(frames, frame_indices, strict=True)
    )
    return _DecodedVideo(
        frames=prepared,
        frame_indices=frame_indices,
        frames_per_second=frames_per_second,
        duration_seconds=(frame_indices[-1] + 1) / frames_per_second,
    )


def _official_video_patch_layout(
    frames: np.ndarray,
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> np.ndarray:
    if padding := -len(frames) % temporal_patch_size:
        frames = np.concatenate((frames, np.repeat(frames[-1:], padding, axis=0)))
    frame_count, channels, height, width = frames.shape
    grid_time = frame_count // temporal_patch_size
    grid_height = height // patch_size
    grid_width = width // patch_size
    return frames.reshape(
        grid_time,
        temporal_patch_size,
        channels,
        grid_height // merge_size,
        merge_size,
        patch_size,
        grid_width // merge_size,
        merge_size,
        patch_size,
    ).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
        grid_time * grid_height * grid_width,
        channels * temporal_patch_size * patch_size * patch_size,
    )


def test_qwen4_exp_video_matches_official_sampling_layout_and_timestamps() -> None:
    class TinyProcessor(Qwen4ExpVisionProcessor):
        patch_size = 2
        minimum_video_pixels = 0
        maximum_video_pixels = 1_000_000

    raw_frames = [
        np.full((4, 8, 3), (index * 31, index * 17, index * 7), dtype=np.uint8)
        for index in range(3)
    ]
    decoded = _decoded_video(
        raw_frames,
        frame_indices=(0, 3, 6),
        frames_per_second=3.0,
    )

    actual, grid, placeholder = TinyProcessor._prepare_video(decoded)
    normalized = np.stack(
        [frame.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0 for frame in raw_frames]
    )
    reference = _official_video_patch_layout(
        normalized,
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
    )

    assert grid == (2, 2, 4)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-7)
    assert placeholder == (
        "<0.5 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        "<2.0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
    )
    expected_indices = np.linspace(0, 299, 20).round().astype(np.int64)
    np.testing.assert_array_equal(
        TinyProcessor._sample_video_indices(300, 30.0, 10.0),
        expected_indices,
    )


def test_glm5_next_video_matches_official_sampling_layout_and_timestamps() -> None:
    class TinyProcessor(Glm5NextVisionProcessor):
        patch_size = 2
        minimum_tokens = 0
        maximum_video_tokens = 1_000_000
        image_mean = (0.0, 0.0, 0.0)
        image_std = (1.0, 1.0, 1.0)

    raw_frames = [
        np.full((4, 8, 3), (index * 29, index * 13, index * 5), dtype=np.uint8)
        for index in range(4)
    ]
    decoded = _decoded_video(
        raw_frames,
        frame_indices=(0, 2, 4, 6),
        frames_per_second=2.0,
    )

    actual, grid, placeholder = TinyProcessor._prepare_video(decoded)
    normalized = np.stack(
        [frame.astype(np.float32).transpose(2, 0, 1) / 255.0 for frame in raw_frames]
    )
    reference = _official_video_patch_layout(
        normalized,
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
    )

    assert grid == (2, 2, 4)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-7)
    assert placeholder == (
        "<|begin_of_video|>"
        "<|begin_of_image|><|image|><|image|><|end_of_image|>0.0 seconds"
        "<|begin_of_image|><|image|><|image|><|end_of_image|>2.0 seconds"
        "<|end_of_video|>"
    )
    assert TinyProcessor._sample_video_indices(31, 10.0, 3.1) == (
        0,
        5,
        10,
        15,
        20,
        25,
    )


def test_flash_next_mixed_image_video_preserves_media_order_and_binary_transport() -> None:
    class TinyProcessor(Qwen4ExpVisionProcessor):
        patch_size = 2
        minimum_pixels = 0
        maximum_pixels = 1_000_000
        minimum_video_pixels = 0
        maximum_video_pixels = 1_000_000

    image_pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    video_frames = [
        np.full((4, 8, 3), index * 40, dtype=np.uint8) for index in range(3)
    ]
    decoded = _decoded_video(
        video_frames,
        frame_indices=(0, 3, 6),
        frames_per_second=3.0,
    )
    processor = TinyProcessor()
    processor._decode_video_for_request = lambda data: decoded
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(Image.fromarray(image_pixels))},
                },
                {"type": "text", "text": " Then watch: "},
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,AA=="},
                },
            ],
        }
    ]

    encoded = processor.prepare_openai_messages(messages)
    binary = processor.prepare_openai_messages(messages, use_binary_file=True)
    assert encoded is not None and binary is not None
    assert encoded.messages == binary.messages
    assert (encoded.source_count, encoded.frame_count) == (2, 3)
    np.testing.assert_array_equal(
        _decode_tensor(encoded.tensors["vision_types"]),
        [1, 2],
    )
    np.testing.assert_array_equal(
        _decode_tensor(encoded.tensors["vision_grid_thw"]),
        [[1, 2, 4], [2, 2, 4]],
    )
    np.testing.assert_array_equal(
        _decode_tensor(encoded.tensors["image_grid_thw"]),
        [[1, 2, 4]],
    )
    np.testing.assert_array_equal(
        _decode_tensor(encoded.tensors["video_grid_thw"]),
        [[2, 2, 4]],
    )
    assert encoded.messages[0]["content"].index("<|image_pad|>") < encoded.messages[0][
        "content"
    ].index("<|video_pad|>")
    try:
        for name in (
            "pixel_values",
            "vision_grid_thw",
            "vision_types",
            "image_grid_thw",
            "video_grid_thw",
        ):
            np.testing.assert_array_equal(
                _decode_binary_tensor(binary.tensors, name),
                _decode_tensor(encoded.tensors[name]),
            )
    finally:
        for path in binary.cleanup_paths:
            path.unlink(missing_ok=True)


def test_multimodal_processor_registry_is_architecture_specific() -> None:
    assert isinstance(
        multimodal_processor_for_architecture("MiniCPMO"),
        MiniCPMO45VisionProcessor,
    )
    assert isinstance(
        multimodal_processor_for_architecture("deepseek-v4"),
        DeepseekV4VisionProcessor,
    )
    assert isinstance(
        multimodal_processor_for_architecture("deepseek-v41-vision"),
        DeepseekV41VisionProcessor,
    )
    assert isinstance(
        multimodal_processor_for_architecture("qwen4_exp"),
        Qwen4ExpVisionProcessor,
    )
    assert isinstance(
        multimodal_processor_for_architecture("glm5-next"),
        Glm5NextVisionProcessor,
    )
    assert isinstance(
        multimodal_processor_for_architecture("qwen3_5"),
        Qwen35VisionProcessor,
    )


def test_image_request_matches_official_slice_and_tensor_contract() -> None:
    width, height = 1200, 600
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 127, dtype=np.uint8),
        ],
        axis=-1,
    )
    image = Image.fromarray(pixels, mode="RGB")
    processor = MiniCPMO45VisionProcessor()
    result = processor.prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
    )

    assert result is not None
    assert result.source_count == 1
    assert result.frame_count == 0
    content = result.messages[0]["content"]
    assert content.count("<image>") == 1
    assert content.count("<slice>") == 3
    assert content.count("<unk>") == 4 * 64
    pixels_tensor = _decode_tensor(result.tensors["pixel_values"])
    mask = _decode_tensor(result.tensors["patch_mask"])
    sizes = _decode_tensor(result.tensors["target_sizes"])
    assert pixels_tensor.shape[0:3] == (4, 3, 14)
    assert mask.shape[0] == 4
    assert sizes.shape == (4, 2)
    np.testing.assert_array_equal(mask.sum(axis=1), sizes.prod(axis=1))
    assert pixels_tensor.dtype == np.dtype("<f4")
    assert np.isfinite(pixels_tensor).all()
    assert pixels_tensor.min() >= -1.0
    assert pixels_tensor.max() <= 1.0


def test_video_request_samples_frames_and_uses_unsliced_placeholders() -> None:
    av = pytest.importorskip("av")
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("h264", rate=2)
        stream.width = 56
        stream.height = 28
        stream.pix_fmt = "yuv420p"
        for index in range(6):
            image = np.full((28, 56, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")

    reference: list[np.ndarray] = []
    next_timestamp = 0.0
    with av.open(io.BytesIO(output.getvalue()), mode="r") as container:
        stream = next(item for item in container.streams if item.type == "video")
        average_rate = float(stream.average_rate) if stream.average_rate else 30.0
        for index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.time) if frame.time is not None else index / max(average_rate, 1.0)
            )
            if reference and timestamp + 1.0e-9 < next_timestamp:
                continue
            reference.append(np.asarray(frame.to_image().convert("RGB")))
            next_timestamp = timestamp + 1.0
    optimized = MiniCPMO45VisionProcessor._decode_video(output.getvalue())
    assert len(optimized) == len(reference)
    for actual, expected in zip(optimized, reference, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)
    prepared = MiniCPMO45VisionProcessor()._decode_video_for_request(output.getvalue())
    assert [frame.source_size for frame in prepared] == [
        (image.shape[1], image.shape[0]) for image in reference
    ]
    for actual, expected in zip(prepared, reference, strict=True):
        resized = MiniCPMO45VisionProcessor._resize_video_frame(
            Image.fromarray(expected, mode="RGB")
        )
        np.testing.assert_array_equal(np.asarray(actual.image), np.asarray(resized))

    result = MiniCPMO45VisionProcessor().prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
                    }
                ],
            }
        ]
    )

    assert result is not None
    assert result.source_count == 1
    assert 2 <= result.frame_count <= 4
    content = result.messages[0]["content"]
    assert content.count("<image>") == result.frame_count
    assert "<slice>" not in content
    assert _decode_tensor(result.tensors["pixel_values"]).shape[0] == result.frame_count


def test_audio_request_builds_exact_mel_and_placeholder_contract() -> None:
    sample_rate = 16_000
    samples = np.arange(sample_rate, dtype=np.float32)
    waveform = np.sin(samples * np.float32(2.0 * np.pi * 220.0 / sample_rate))
    pcm = np.rint(waveform * np.float32(12_000.0)).astype("<i2")
    encoded_audio = io.BytesIO()
    with wave.open(encoded_audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    encoded = base64.b64encode(encoded_audio.getvalue()).decode("ascii")

    result = MiniCPMO45VisionProcessor().prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav"},
                    },
                    {"type": "text", "text": "Please answer the recording."},
                ],
            }
        ]
    )

    assert result is not None
    assert result.source_count == 1
    assert result.frame_count == 0
    content = result.messages[0]["content"]
    features = _decode_tensor(result.tensors["audio_features"])
    lengths = _decode_tensor(result.tensors["audio_lengths"])
    assert features.shape == (1, 80, 100)
    np.testing.assert_array_equal(lengths, [100])
    pooled = (((int(lengths[0]) - 1) // 2 + 1) - 5) // 5 + 1
    assert content.startswith("<|audio_start|>")
    assert content.count("<unk>") == pooled
    assert "<|audio_end|>\nPlease answer the recording." in content
    assert np.isfinite(features).all()


def test_binary_tensor_transport_matches_base64_and_uses_private_file() -> None:
    class TinyProcessor(MiniCPMO45VisionProcessor):
        scale_resolution = 28
        maximum_image_slices = 1

    image = Image.new("RGB", (35, 21), (20, 40, 60))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _data_url(image)}}],
        }
    ]
    processor = TinyProcessor()
    encoded = processor.prepare_openai_messages(messages)
    binary = processor.prepare_openai_messages(messages, use_binary_file=True)
    assert encoded is not None
    assert binary is not None
    assert encoded.messages == binary.messages
    assert binary.source_count == encoded.source_count
    assert len(binary.cleanup_paths) == 1
    path = binary.cleanup_paths[0]
    try:
        file_spec = binary.tensors["binary_file"]
        assert file_spec["path"] == str(path)
        assert path.stat().st_size == file_spec["size"]
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with path.open("rb") as stream:
            header = stream.read(64)
        assert header[:8] == b"MFQMM01\0"
        assert header[8:40].hex() == file_spec["token"]
        for name in ("pixel_values", "patch_mask", "target_sizes"):
            np.testing.assert_array_equal(
                _decode_binary_tensor(binary.tensors, name),
                _decode_tensor(encoded.tensors[name]),
            )
    finally:
        path.unlink(missing_ok=True)


def test_backend_sends_shared_vision_tensor_protocol_to_native_worker() -> None:
    captured: dict[str, object] = {}

    class TinyProcessor(MiniCPMO45VisionProcessor):
        scale_resolution = 28
        maximum_image_slices = 1

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"model": "MiniCPM-o", "model_type": "minicpmo"},
            )
        captured["payload"] = json.loads(request.content)
        body = (
            'data: {"choices":[{"delta":{"content":"ok"},'
            '"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        backend._vision_processor = TinyProcessor()
        image = Image.new("RGB", (28, 28), (20, 40, 60))
        deltas = [
            item
            async for item in backend.stream(
                model="MiniCPM-o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url(image)},
                            }
                        ],
                    }
                ],
                sampling=SamplingParams(max_tokens=4),
            )
        ]
        await client.aclose()
        assert deltas[0].content_delta == "ok"

    asyncio.run(run())
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"].startswith("<image_id>0</image_id>")
    tensors = payload["mfq_multimodal"]
    assert tensors["version"] == 1
    assert tensors["pixel_values"]["shape"] == [1, 3, 14, 56]
    assert tensors["patch_mask"]["shape"] == [1, 4]
    assert tensors["target_sizes"]["shape"] == [1, 2]


def test_backend_cleans_local_binary_tensor_file_after_stream() -> None:
    captured_path = None

    class TinyProcessor(MiniCPMO45VisionProcessor):
        scale_resolution = 28
        maximum_image_slices = 1

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_path
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"model": "MiniCPM-o", "model_type": "minicpmo"},
            )
        payload = json.loads(request.content)
        captured_path = payload["mfq_multimodal"]["binary_file"]["path"]
        assert Path(captured_path).is_file()
        body = (
            'data: {"choices":[{"delta":{"content":"ok"},'
            '"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend(
            "http://backend",
            client=client,
            local_tensor_files=True,
        )
        backend._vision_processor = TinyProcessor()
        image = Image.new("RGB", (28, 28), (20, 40, 60))
        deltas = [
            item
            async for item in backend.stream(
                model="MiniCPM-o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url(image)},
                            }
                        ],
                    }
                ],
                sampling=SamplingParams(max_tokens=4),
            )
        ]
        await client.aclose()
        assert deltas[0].content_delta == "ok"

    asyncio.run(run())
    assert captured_path is not None
    assert not Path(captured_path).exists()


def test_backend_reports_product_level_multimodal_prefill() -> None:
    request_id = "chatcmpl-multimodal"

    class TinyProcessor(MiniCPMO45VisionProcessor):
        scale_resolution = 28
        maximum_image_slices = 1

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"model": "MiniCPM-o", "model_type": "minicpmo"},
            )
        if request.url.path == "/api/status":
            return httpx.Response(
                200,
                json={"model": "MiniCPM-o", "last_request": {"id": request_id}},
            )
        metrics = {
            "prefill_tokens": 4,
            "ttft_ms": 20.0,
            "prefill_ms": 10.0,
            "prefill_tps": 400.0,
            "multimodal_ms": 7.0,
            "model_prefill_ms": 17.0,
            "decode_ms": 2.0,
            "decode_tps": 500.0,
            "generation_ms": 24.0,
            "generation_tps": 125.0,
            "sampling": SamplingParams(max_tokens=4).model_dump(mode="json"),
        }
        body = (
            "data: "
            + json.dumps(
                {
                    "id": request_id,
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "mfq_metrics": metrics,
                }
            )
            + "\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        backend._vision_processor = TinyProcessor()
        image = Image.new("RGB", (28, 28), (20, 40, 60))
        deltas = [
            item
            async for item in backend.stream(
                model="MiniCPM-o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url(image)},
                            }
                        ],
                    }
                ],
                sampling=SamplingParams(max_tokens=4),
            )
        ]
        performance = deltas[0].performance
        assert performance is not None
        assert performance.processor_ms > 0.0
        assert performance.complete_prefill_ms == pytest.approx(
            performance.processor_ms + performance.ttft_ms
        )
        assert performance.complete_prefill_tps == pytest.approx(
            4000.0 / performance.complete_prefill_ms
        )
        assert performance.complete_generation_ms == pytest.approx(
            performance.processor_ms + performance.generation_ms
        )
        status = await backend.runtime_status()
        assert status["last_request"]["processor_ms"] == performance.processor_ms
        assert status["last_request"]["complete_prefill_tps"] == performance.complete_prefill_tps
        await client.aclose()

    asyncio.run(run())
