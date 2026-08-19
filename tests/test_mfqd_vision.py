from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from mfqd.vision import MiniCPMO45VisionProcessor


def _data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_tensor(tensor: dict[str, object]) -> np.ndarray:
    dtypes = {
        "float32": "<f4",
        "int32": "<i4",
        "uint8": "u1",
    }
    raw = base64.b64decode(str(tensor["data_base64"]), validate=True)
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
                float(frame.time)
                if frame.time is not None
                else index / max(average_rate, 1.0)
            )
            if reference and timestamp + 1.0e-9 < next_timestamp:
                continue
            reference.append(np.asarray(frame.to_image().convert("RGB")))
            next_timestamp = timestamp + 1.0
    optimized = MiniCPMO45VisionProcessor._decode_video(output.getvalue())
    assert len(optimized) == len(reference)
    for actual, expected in zip(optimized, reference, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)

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
