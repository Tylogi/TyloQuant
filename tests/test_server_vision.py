from __future__ import annotations

import asyncio
import base64
import io
import json
import stat
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from mfq.server.backend import OpenAIChatBackend
from mfq.server.models import SamplingParams
from mfq.server.vision import MiniCPMO45VisionProcessor


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


def _decode_binary_tensor(tensors: dict[str, object], name: str) -> np.ndarray:
    dtypes = {
        "float32": "<f4",
        "int32": "<i4",
        "uint8": "u1",
    }
    file_spec = tensors["binary_file"]
    tensor = tensors[name]
    assert isinstance(file_spec, dict)
    assert isinstance(tensor, dict)
    with open(str(file_spec["path"]), "rb") as stream:
        stream.seek(int(tensor["data_offset"]))
        raw = stream.read(int(tensor["data_length"]))
    return np.frombuffer(raw, dtype=dtypes[str(tensor["dtype"])]).reshape(
        tuple(tensor["shape"])
    )


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
    prepared = MiniCPMO45VisionProcessor()._decode_video_for_request(
        output.getvalue()
    )
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


def test_binary_tensor_transport_matches_base64_and_uses_private_file() -> None:
    class TinyProcessor(MiniCPMO45VisionProcessor):
        scale_resolution = 28
        maximum_image_slices = 1

    image = Image.new("RGB", (35, 21), (20, 40, 60))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image)}}
            ],
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
        assert (
            status["last_request"]["complete_prefill_tps"]
            == performance.complete_prefill_tps
        )
        await client.aclose()

    asyncio.run(run())
