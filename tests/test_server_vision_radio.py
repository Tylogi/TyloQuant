"""RADIO (cropped-position-embedding) image preprocessing tests.

The reference implementations live here: the resize is compared directly
against ``torch.nn.functional.interpolate`` and, when
``MFQ_NEMOTRON_HF_DIR`` points at the released checkpoint, the whole image
path is compared against that checkpoint's own image processor.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from PIL import Image

from mfq.server.vision import RadioVisionProcessor, multimodal_processor_for_architecture
from mfq.server.vision.decode import VisionProcessingError
from mfq.server.vision.resize import antialiased_bicubic_resize

RESIZE_CASES = (
    # (height, width, target height, target width): shrink, grow, non-integer,
    # non-square and identity in both directions.
    (17, 23, 9, 11),
    (9, 11, 17, 23),
    (600, 1200, 608, 1216),
    (300, 400, 304, 408),
    (100, 100, 101, 101),
    (64, 48, 32, 24),
    (33, 47, 64, 90),
    (48, 48, 48, 17),
    (97, 53, 97, 53),
)

TARGET_PATCH_CASES = (
    # (width, height, patches across, patches down)
    (64, 64, 32, 32),
    (100, 100, 32, 32),
    (33, 17, 40, 28),
    (2048, 1024, 128, 64),
    (4000, 4000, 116, 114),
    (1200, 600, 76, 38),
)


def _data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8))


@pytest.mark.parametrize("size", [(4000, 50), (50, 4000), (4096, 4096), (1, 4096)])
def test_large_and_narrow_images_respect_model_patch_budget(size):
    processor = RadioVisionProcessor()
    image = Image.new("RGB", size)
    patches, (_, rows, columns) = processor._prepare_image(image, processor.tokens_available(1))
    assert rows > 0 and columns > 0
    assert rows % processor.downsample_factor == columns % processor.downsample_factor == 0
    assert rows * columns <= processor.max_num_patches
    assert patches.shape == (rows * columns, 3 * processor.patch_size**2)


def test_unrepresentable_aspect_ratio_is_a_clear_input_error():
    processor = RadioVisionProcessor()
    with pytest.raises(VisionProcessingError, match="aspect ratio.*patch budget"):
        processor.target_patches(1_000_000, 1, processor.tokens_available(1))


def test_antialiased_bicubic_matches_torch() -> None:
    rng = np.random.default_rng(11)
    for height, width, target_height, target_width in RESIZE_CASES:
        pixels = (rng.random((3, height, width)) * 255.0).astype(np.float32)
        expected = functional.interpolate(
            torch.from_numpy(pixels)[None],
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0].numpy()
        actual = antialiased_bicubic_resize(pixels, target_height, target_width)
        assert actual.shape == (3, target_height, target_width)
        # torch accumulates in float32; the remaining difference is rounding.
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=0.0)
        # Normalised pixels (the values the tower sees) stay within 1e-5.
        np.testing.assert_allclose(actual / 255.0, expected / 255.0, atol=1e-5, rtol=0.0)


def test_antialiased_bicubic_identity_is_exact() -> None:
    rng = np.random.default_rng(5)
    pixels = rng.normal(0, 1, (3, 12, 9)).astype(np.float32)
    np.testing.assert_array_equal(antialiased_bicubic_resize(pixels, 12, 9), pixels)


@pytest.mark.parametrize(
    ("width", "height", "patches_across", "patches_down"),
    TARGET_PATCH_CASES,
)
def test_target_patch_rule(
    width: int,
    height: int,
    patches_across: int,
    patches_down: int,
) -> None:
    processor = RadioVisionProcessor()
    budget = processor.tokens_available(1)
    assert budget == processor.max_num_patches
    assert processor.target_patches(width, height, budget) == (
        patches_across,
        patches_down,
    )
    # Every side is a multiple of the pixel-shuffle factor and fits the budget.
    assert patches_across % processor.downsample_factor == 0
    assert patches_down % processor.downsample_factor == 0
    assert patches_across * patches_down <= budget


def test_target_patch_rule_respects_minimum_and_cap() -> None:
    processor = RadioVisionProcessor(min_num_patches=64, max_num_patches=256)
    budget = processor.tokens_available(1)
    assert budget == 256
    # A tiny image is scaled up until it reaches the per-image minimum.
    across, down = processor.target_patches(64, 64, budget)
    assert across * down >= 64
    # A large image is scaled down to the cap, never past it.
    across, down = processor.target_patches(1600, 1600, budget)
    assert across * down <= 256


def test_processor_registry_selects_radio_for_nemotron() -> None:
    for model_type in ("nemotron_h", "nemotron_h_text", "nemotron_h_omni", "Nemotron-H"):
        processor = multimodal_processor_for_architecture(model_type)
        assert isinstance(processor, RadioVisionProcessor)
        assert processor.input_contract == "radio_cpe.v1"


def test_placeholder_and_patch_tensors_match_pixel_shuffle_tokens() -> None:
    processor = RadioVisionProcessor()
    images = [_image(300, 180, seed=1), _image(96, 96, seed=2)]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare "},
                {"type": "image_url", "image_url": {"url": _data_url(images[0])}},
                {"type": "text", "text": " with "},
                {"type": "image_url", "image_url": {"url": _data_url(images[1])}},
            ],
        }
    ]
    processed = processor.prepare_openai_messages(messages)
    assert processed is not None
    assert processed.source_count == 2
    assert processed.frame_count == 0
    content = processed.messages[0]["content"]
    assert content.startswith("compare <img>")
    assert content.count("<img>") == content.count("</img>") == 2
    groups = []
    for block in content.split("<img>")[1:]:
        placeholder, tail = block.split("</img>")
        assert tail in {" with ", ""}
        assert len(placeholder) % len("<image>") == 0
        assert placeholder == "<image>" * (len(placeholder) // len("<image>"))
        groups.append(len(placeholder) // len("<image>"))
    tensors = processed.tensors
    assert tensors["processor"] == "radio_cpe.v1"
    assert tensors["version"] == 3
    assert tensors["vision_types"]["shape"] == [2]
    grids = _decode(tensors["vision_grid_thw"])
    np.testing.assert_array_equal(grids[:, 0], [1, 1])
    patch_width = 3 * processor.patch_size * processor.patch_size
    assert tensors["pixel_values"]["shape"] == [
        int(np.sum(grids[:, 1] * grids[:, 2])),
        patch_width,
    ]
    # The placeholder count is the pixel-shuffled patch count of every image.
    assert groups == [int(grid[1] * grid[2]) // 4 for grid in grids]


def test_pixels_are_normalised_exactly_once() -> None:
    processor = RadioVisionProcessor()
    image = _image(160, 96, seed=4)
    processed = processor.prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": _data_url(image)}}],
            }
        ]
    )
    assert processed is not None
    pixels = _decode(processed.tensors["pixel_values"])
    _, rows, columns = _decode(processed.tensors["vision_grid_thw"])[0]
    target_height = rows * processor.patch_size
    target_width = columns * processor.patch_size
    raw = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)
    resized = antialiased_bicubic_resize(raw, target_height, target_width)
    mean = np.asarray(processor.image_mean, dtype=np.float32)[:, None, None]
    deviation = np.asarray(processor.image_std, dtype=np.float32)[:, None, None]
    expected = (resized / np.float32(255.0) - mean) / deviation
    expected = expected.reshape(3, rows, processor.patch_size, columns, processor.patch_size)
    expected = expected.transpose(1, 3, 0, 2, 4).reshape(rows * columns, -1)
    np.testing.assert_allclose(pixels, expected, atol=1e-6, rtol=0.0)
    # Resizing after normalisation, or normalising a second time, changes the
    # values well beyond the tolerance above: the assertion is discriminating.
    twice = (expected * deviation + mean) / 255.0
    twice = (twice - mean) / deviation
    assert float(np.max(np.abs(twice - expected))) > 1e-3


def test_media_types_outside_the_contract_are_rejected() -> None:
    processor = RadioVisionProcessor()
    for media in ("video_url", "input_video", "audio_url"):
        with pytest.raises(VisionProcessingError, match="images only"):
            processor.prepare_openai_messages(
                [{"role": "user", "content": [{"type": media, media: {"url": "data:,"}}]}]
            )
    for token in ("<img>", "<image>", "</img>"):
        with pytest.raises(VisionProcessingError, match="cannot be supplied as text"):
            processor.prepare_openai_messages(
                [{"role": "user", "content": [{"type": "text", "text": token}]}]
            )


def test_binary_transport_matches_inline_tensors() -> None:
    processor = RadioVisionProcessor()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _data_url(_image(80, 48, 9))}}],
        }
    ]
    inline = processor.prepare_openai_messages(messages)
    binary = processor.prepare_openai_messages(messages, use_binary_file=True)
    assert inline is not None and binary is not None
    assert inline.messages == binary.messages
    try:
        for name in ("pixel_values", "vision_grid_thw", "vision_types", "image_grid_thw"):
            np.testing.assert_array_equal(
                _decode_binary(binary.tensors, name),
                _decode(inline.tensors[name]),
            )
    finally:
        for path in binary.cleanup_paths:
            path.unlink(missing_ok=True)


def _decode(tensor: dict[str, object]) -> np.ndarray:
    dtype = {
        "float32": "<f4",
        "int32": "<i4",
        "int64": "<i8",
        "uint8": "u1",
    }[str(tensor["dtype"])]
    values = np.frombuffer(base64.b64decode(tensor["data_base64"]), dtype=dtype)
    return values.reshape(tuple(int(item) for item in tensor["shape"]))


def _decode_binary(tensors: dict[str, object], name: str) -> np.ndarray:
    path = Path(str(tensors["binary_file"]["path"]))
    shape = tuple(int(item) for item in tensors[name]["shape"])
    dtype = {
        "float32": "<f4",
        "int32": "<i4",
        "int64": "<i8",
        "uint8": "u1",
    }[str(tensors[name]["dtype"])]
    offset = int(tensors[name]["data_offset"])
    length = int(tensors[name]["data_length"])
    with path.open("rb") as stream:
        stream.seek(offset)
        return np.frombuffer(stream.read(length), dtype=dtype).reshape(shape)


def _checkpoint_dir() -> Path:
    value = os.environ.get("MFQ_NEMOTRON_HF_DIR")
    if not value:
        pytest.skip("MFQ_NEMOTRON_HF_DIR is not set")
    path = Path(value)
    if not (path / "image_processing_nemotron_h_omni.py").is_file():
        pytest.skip("MFQ_NEMOTRON_HF_DIR has no Nemotron-H image processor")
    return path


def _hf_image_processor(directory: Path):
    spec = importlib.util.spec_from_file_location(
        "mfq_hf_nemotron_image_processing",
        directory / "image_processing_nemotron_h_omni.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    configuration = json.loads((directory / "preprocessor_config.json").read_text())
    configuration.pop("auto_map", None)
    configuration.pop("image_processor_type", None)
    return module.NemotronH_Omni_Reasoning_V3ImageProcessor(**configuration)


def test_image_processor_matches_the_checkpoint_processor() -> None:
    directory = _checkpoint_dir()
    reference = _hf_image_processor(directory)
    processor = RadioVisionProcessor()
    for width, height in ((1200, 600), (64, 64), (33, 17), (1024, 768), (38, 90)):
        image = _image(width, height, seed=width)
        inputs = reference(images=[image], return_tensors="pt")
        expected_pixels = inputs["pixel_values"][0].numpy()
        columns, rows = processor.target_patches(width, height, processor.tokens_available(1))
        assert int(inputs["num_tokens"][0]) == rows * columns // 4
        actual_patches, grid = processor._prepare_image(image, processor.tokens_available(1))
        assert grid == (1, rows, columns)
        patch = processor.patch_size
        expected = expected_pixels.reshape(3, rows, patch, columns, patch)
        expected = expected.transpose(1, 3, 0, 2, 4).reshape(rows * columns, -1)
        np.testing.assert_allclose(actual_patches, expected, atol=1e-5, rtol=0.0)


def test_cleanup_paths_are_reported_for_binary_transport() -> None:
    processor = RadioVisionProcessor()
    processed = processor.prepare_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(_image(40, 40))}}
                ],
            }
        ],
        use_binary_file=True,
    )
    assert processed is not None
    assert len(processed.cleanup_paths) == 1
    path = processed.cleanup_paths[0]
    assert path.is_file()
    assert Path(processed.tensors["binary_file"]["path"]) == path
    path.unlink()
    assert not path.exists()
