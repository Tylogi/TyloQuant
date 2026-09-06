from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.runtime.mlx_flash_next_vision import (  # noqa: E402
    MlxGlm5NextVision,
    MlxQwen4ExpVision,
    inject_vision_embeddings,
    qwen4_multimodal_positions,
    vision_layout,
)
from mfq.runtime.mlx_linear import MlxNintModel  # noqa: E402


def _random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 0.05,
) -> np.ndarray:
    return rng.normal(scale=scale, size=shape).astype(np.float32)


def _qwen_vision_fixture() -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    rng = np.random.default_rng(3810)
    hidden, intermediate, output = 8, 12, 6
    config = SimpleNamespace(
        model_type="qwen4_exp_vision",
        hidden_size=hidden,
        intermediate_size=intermediate,
        depth=1,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=output,
        hidden_act="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        swiglu_limit=10.0,
        image_size=None,
        projection_intermediate_size=None,
        num_position_embeddings=16,
    )
    prefix = "model.visual"
    tensors = {
        prefix + ".patch_embed.proj.weight": _random(rng, (hidden, 3, 1, 2, 2)),
        prefix + ".patch_embed.proj.bias": _random(rng, (hidden,)),
        prefix + ".pos_embed.weight": _random(rng, (16, hidden)),
        prefix + ".blocks.0.attn.qkv.weight": _random(rng, (3 * hidden, hidden)),
        prefix + ".blocks.0.attn.qkv.bias": _random(rng, (3 * hidden,)),
        prefix + ".blocks.0.attn.proj.weight": _random(rng, (hidden, hidden)),
        prefix + ".blocks.0.attn.proj.bias": _random(rng, (hidden,)),
        prefix + ".blocks.0.norm1.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".blocks.0.norm1.bias": np.zeros((hidden,), dtype=np.float32),
        prefix + ".blocks.0.norm2.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".blocks.0.norm2.bias": np.zeros((hidden,), dtype=np.float32),
        prefix + ".blocks.0.mlp.linear_fc1.weight": _random(
            rng, (intermediate, hidden)
        ),
        prefix + ".blocks.0.mlp.linear_fc1.bias": _random(rng, (intermediate,)),
        prefix + ".blocks.0.mlp.linear_fc2.weight": _random(
            rng, (hidden, intermediate)
        ),
        prefix + ".blocks.0.mlp.linear_fc2.bias": _random(rng, (hidden,)),
        prefix + ".merger.norm.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".merger.norm.bias": np.zeros((hidden,), dtype=np.float32),
        prefix + ".merger.linear_fc1.weight": _random(
            rng, (4 * hidden, 4 * hidden)
        ),
        prefix + ".merger.linear_fc1.bias": _random(rng, (4 * hidden,)),
        prefix + ".merger.linear_fc2.weight": _random(rng, (output, 4 * hidden)),
        prefix + ".merger.linear_fc2.bias": _random(rng, (output,)),
    }
    return SimpleNamespace(vision=config), tensors


def _glm_vision_fixture() -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    rng = np.random.default_rng(5310)
    hidden, intermediate, output, projection = 8, 12, 16, 20
    config = SimpleNamespace(
        model_type="glm5_next_vision",
        hidden_size=hidden,
        intermediate_size=intermediate,
        depth=1,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=output,
        hidden_act="silu",
        attention_bias=True,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        swiglu_limit=7.0,
        image_size=4,
        projection_intermediate_size=projection,
        num_position_embeddings=None,
    )
    prefix = "model.visual"
    tensors = {
        prefix + ".patch_embed.proj.weight": _random(rng, (hidden, 3, 1, 2, 2)),
        prefix + ".patch_embed.proj.bias": _random(rng, (hidden,)),
        prefix + ".blocks.0.attn.qkv.weight": _random(rng, (3 * hidden, hidden)),
        prefix + ".blocks.0.attn.qkv.bias": _random(rng, (3 * hidden,)),
        prefix + ".blocks.0.attn.proj.weight": _random(rng, (hidden, hidden)),
        prefix + ".blocks.0.attn.proj.bias": _random(rng, (hidden,)),
        prefix + ".blocks.0.attn.q_norm.weight": np.ones((4,), dtype=np.float32),
        prefix + ".blocks.0.attn.k_norm.weight": np.ones((4,), dtype=np.float32),
        prefix + ".blocks.0.norm1.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".blocks.0.norm2.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".blocks.0.mlp.gate_proj.weight": _random(
            rng, (intermediate, hidden)
        ),
        prefix + ".blocks.0.mlp.gate_proj.bias": _random(rng, (intermediate,)),
        prefix + ".blocks.0.mlp.up_proj.weight": _random(rng, (intermediate, hidden)),
        prefix + ".blocks.0.mlp.up_proj.bias": _random(rng, (intermediate,)),
        prefix + ".blocks.0.mlp.down_proj.weight": _random(
            rng, (hidden, intermediate)
        ),
        prefix + ".blocks.0.mlp.down_proj.bias": _random(rng, (hidden,)),
        prefix + ".post_layernorm.weight": np.ones((hidden,), dtype=np.float32),
        prefix + ".downsample.weight": _random(rng, (output, hidden, 2, 2)),
        prefix + ".downsample.bias": _random(rng, (output,)),
        prefix + ".merger.proj.weight": _random(rng, (output, output)),
        prefix + ".merger.post_projection_norm.weight": np.ones(
            (output,), dtype=np.float32
        ),
        prefix + ".merger.post_projection_norm.bias": np.zeros(
            (output,), dtype=np.float32
        ),
        prefix + ".merger.gate_proj.weight": _random(rng, (projection, output)),
        prefix + ".merger.up_proj.weight": _random(rng, (projection, output)),
        prefix + ".merger.down_proj.weight": _random(rng, (output, projection)),
    }
    return SimpleNamespace(vision=config), tensors


def test_vision_layout_uses_spatial_merge_block_order_and_frame_segments() -> None:
    positions, lengths = vision_layout(np.asarray([[2, 2, 4]], dtype=np.int32), 2)
    expected_frame = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(positions, np.tile(expected_frame, (2, 1)))
    assert lengths == (8, 8)


@pytest.mark.parametrize(
    ("fixture", "runtime"),
    [
        (_qwen_vision_fixture, MlxQwen4ExpVision),
        (_glm_vision_fixture, MlxGlm5NextVision),
    ],
)
def test_flash_next_vision_packed_items_match_independent_execution(
    fixture,
    runtime,
) -> None:
    config, tensors = fixture()
    model = runtime(MlxNintModel(tensors), config)
    patch_width = 3 * 1 * 2 * 2
    rng = np.random.default_rng(99)
    first_pixels = _random(rng, (4, patch_width), 0.2)
    second_pixels = _random(rng, (4, patch_width), 0.2)
    grid = np.asarray([[1, 2, 2]], dtype=np.int32)

    first_hidden, first_merged = model(first_pixels, grid)
    second_hidden, second_merged = model(second_pixels, grid)
    packed_hidden, packed_merged = model(
        np.concatenate((first_pixels, second_pixels), axis=0),
        np.asarray([[1, 2, 2], [1, 2, 2]], dtype=np.int32),
    )
    expected_hidden = mx.concatenate((first_hidden, second_hidden), axis=0)
    expected_merged = mx.concatenate((first_merged, second_merged), axis=0)
    mx.eval(packed_hidden, packed_merged, expected_hidden, expected_merged)

    np.testing.assert_allclose(
        np.asarray(packed_hidden),
        np.asarray(expected_hidden),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(packed_merged),
        np.asarray(expected_merged),
        rtol=2e-3,
        atol=2e-3,
    )


def test_inject_vision_embeddings_replaces_image_and_video_placeholders() -> None:
    token_embeddings = mx.zeros((1, 5, 3), dtype=mx.float16)
    input_ids = mx.array([[9, 41, 8, 42, 7]], dtype=mx.int32)
    vision = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float16)
    actual = inject_vision_embeddings(
        token_embeddings,
        input_ids,
        vision,
        (41, 42),
    )
    mx.eval(actual)
    expected = np.zeros((1, 5, 3), dtype=np.float16)
    expected[0, 1] = (1, 2, 3)
    expected[0, 3] = (4, 5, 6)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_qwen4_multimodal_positions_match_text_image_and_video_frame_layout() -> None:
    # Text(2), one 2x4 image after merge(2), text(1), and two 2x2 video frames.
    ids = np.zeros((1, 10), dtype=np.int32)
    modalities = np.asarray([[0, 0, 1, 1, 0, 2, 0, 2, 0, 0]], dtype=np.int32)
    positions, deltas = qwen4_multimodal_positions(
        ids,
        modalities,
        spatial_merge_size=2,
        image_grid_thw=np.asarray([[1, 2, 4]], dtype=np.int32),
        video_grid_thw=np.asarray([[2, 2, 2]], dtype=np.int32),
    )
    expected = np.asarray(
        [
            [0, 1, 2, 2, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 2, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ],
        dtype=np.int32,
    )[:, None]
    np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(deltas, np.zeros((1, 1), dtype=np.int32))
