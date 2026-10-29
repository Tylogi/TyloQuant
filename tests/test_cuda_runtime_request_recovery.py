"""A failed RADIO request must leave the shared runtime able to serve text."""

import asyncio
import os

import numpy as np
import pytest
from PIL import Image

from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from mfq.server.vision.decode import _tensor
from mfq.server.vision.radio import RadioVisionProcessor
from tests.test_cuda_nemotron_vision import PATCH_WIDTH, RADIO_COMPONENT, vision_fixture
from tests.test_cuda_nemotron_runtime import model_fixture, write_nemotron_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("context", [32, 4096])
def test_radio_invalid_image_then_text_and_context_boundary(tmp_path, context):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("requires a native CUDA runtime")
    config, weights = vision_fixture()
    text_config, text_weights = model_fixture(("mamba", "moe", "attention"))
    config["llm_config"] = text_config["llm_config"]
    config["llm_config"]["max_position_embeddings"] = context
    weights.update(text_weights)
    model = tmp_path / "radio-recovery.mfq"
    write_nemotron_fixture(model, config, weights, extra_components=(RADIO_COMPONENT,))
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)

    async def run():
        process = await asyncio.create_subprocess_exec(
            executable, "--model", str(model), "--tokenizer", str(tokenizer),
            "--transport", "stdio", "--ctx-size", str(context),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        client = StdioRuntimeClient(process)
        try:
            async def generate(prompt, media=None, max_tokens=1):
                payload = dict(messages=[dict(role="user", content="test")],
                    mfq_preformatted_prompt=prompt, max_tokens=max_tokens, temperature=0,
                    enable_mtp=False, stream=True, stream_options={"include_usage": True},
                    mfq_prefix_cache_enabled=True)
                if media is not None:
                    payload["mfq_multimodal"] = media
                async with client.generate(payload) as events:
                    return [event async for event in events if event]

            media = dict(version=3, processor="radio_cpe.v1", **{
                "pixel_values": _tensor(np.zeros((16, PATCH_WIDTH), np.float32), "float32"),
                "vision_grid_thw": _tensor(np.array([[1, 4, 4]], np.int32), "int32"),
                "image_grid_thw": _tensor(np.array([[1, 4, 4]], np.int32), "int32"),
                "vision_types": _tensor(np.array([1], np.int32), "int32"),
            })
            # Geometry is valid but the prompt has too few image placeholders;
            # this fails in preparation on the scheduler's execution thread.
            with pytest.raises(BackendError) as caught:
                await generate("token5token18token6", media)
            assert caught.value.status_code == 400
            assert (await client.health())["status"] == "ok"
            assert await generate("token5token6")
            # The position table has side 8; HF permits larger interpolated
            # grids, including this narrow image with columns greater than 8.
            media["vision_grid_thw"] = media["image_grid_thw"] = _tensor(
                np.array([[1, 2, 16]], np.int32), "int32")
            media["pixel_values"] = _tensor(np.zeros((32, PATCH_WIDTH), np.float32), "float32")
            assert await generate("token5" + "token18" * 8 + "token6", media)
            if context == 4096:
                processor = RadioVisionProcessor(patch_size=2)
                for size in [(4000, 50), (4096, 4096)]:
                    pixels, grid = processor._prepare_image(
                        Image.new("RGB", size), processor.tokens_available(1))
                    media["vision_grid_thw"] = media["image_grid_thw"] = _tensor(
                        np.array([grid], np.int32), "int32")
                    media["pixel_values"] = _tensor(pixels, "float32")
                    assert await generate("token5" + "token18" * (grid[1] * grid[2] // 4) + "token6", media)
                    assert await generate("token5token6")
            # Repeated near-full prompts followed by a legal exact boundary.
            for count in [context - 7, context - 5, context - 3]:
                assert await generate("token5" * count)
            events = await generate("token5" * (context - 4), max_tokens=4)
            assert any(event.get("usage", {}).get("prompt_tokens") == context - 4 for event in events)
        finally:
            await client.aclose()
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), timeout=10)

    asyncio.run(run())
