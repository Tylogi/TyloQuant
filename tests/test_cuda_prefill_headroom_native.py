"""Primary expert arenas must yield VRAM before text and media preparation."""
import asyncio
from contextlib import suppress
import os

import numpy as np
import pytest

from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.mx import MxTensor
from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from tests.test_cuda_nemotron_runtime import model_fixture, write_nemotron_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("media", [False, True], ids=["4096-token-prefill", "radio-image"])
def test_prefill_reclaims_primary_experts_under_vram_cap(tmp_path, media):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    components = ()
    if media:
        from tests.test_cuda_nemotron_vision import RADIO_COMPONENT, vision_fixture

        config, weights = vision_fixture()
        config["vision_config"]["max_img_size"] = 64
        weights["vision.position_embedding.weight"] = np.tile(
            weights["vision.position_embedding.weight"], (1, 16, 1))
        components = (RADIO_COMPONENT,)
    else:
        config, weights = model_fixture(("mamba", "moe"))
    experts, intermediate, latent = 1024, 1024, 64
    config["llm_config"].update(max_position_embeddings=8192, n_routed_experts=experts,
        moe_intermediate_size=intermediate, moe_latent_size=latent)
    prefix = "model.block.1.mlp."
    weights[prefix + "router.weight"] = np.tile(weights[prefix + "router.weight"][:1], (experts, 1))
    weights[prefix + "router.bias"] = np.linspace(-0.01, 0.01, experts, dtype=np.float32)
    weights[prefix + "latent.down.weight"] = np.tile(weights[prefix + "latent.down.weight"], (4, 1))
    weights[prefix + "latent.up.weight"] = np.tile(weights[prefix + "latent.up.weight"], (1, 4)) / 4
    for role in ("down", "up"):
        name = prefix + f"latent.{role}.weight"
        weights[name] = weights[name].astype(np.float16)
    for role, rows, width in (("up", intermediate, latent), ("down", latent, intermediate)):
        shape = (experts * rows, width)
        packed = MxTensor("MXFP4", shape, np.full((shape[0], width // 2), 0x22, np.uint8),
            np.full((shape[0], width // 32), 119, np.uint8))
        weights[prefix + f"experts.{role}.weight"] = MfeTensor(
            (experts, rows, width), (MfePool(np.arange(experts, dtype=np.int32), packed),))
    model = tmp_path / "headroom.mfq"
    write_nemotron_fixture(model, config, weights, extra_components=components)
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)
    log_path = tmp_path / "worker.log"

    async def run():
        environment = dict(os.environ, MFQ_CUDA_VRAM_LIMIT_GIB="2",
            MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.046875", MFQ_KV_TURBOQUANT_BITS="0",
            MFQ_MOE_PIPELINE="1", MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_PREFILL_LAYER="1",
            MFQ_MOE_PREFILL_GROUPS="1", MFQ_MOE_HOST_CACHE_BYTES=str(128 << 20),
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1")
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "8192",
                "--prefill-chunk-size", "1024", "--moe-gpu-cache-gb", "0.06",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
            client = StdioRuntimeClient(process, control_timeout_seconds=60)
            try:
                payload = dict(messages=[dict(role="user", content="image" if media else "long")],
                    mfq_preformatted_prompt="token8" * 4096, enable_mtp=False,
                    temperature=0, max_tokens=4, stream=True, stream_options=dict(include_usage=True))
                prompt_tokens = 4096
                if media:
                    from mfq.server.vision.decode import _tensor

                    rows = columns = 28
                    patches = rows * columns
                    prompt_tokens = patches // 4 + 2
                    payload.update(mfq_preformatted_prompt="token8" + "token18" * (patches // 4) + "token9",
                        enable_vision=True, mfq_multimodal=dict(version=3, processor="radio_cpe.v1",
                            pixel_values=_tensor(np.zeros((patches, 12), np.float32), "float32"),
                            vision_grid_thw=_tensor(np.array([[1, rows, columns]], np.int32), "int32"),
                            image_grid_thw=_tensor(np.array([[1, rows, columns]], np.int32), "int32"),
                            vision_types=_tensor(np.array([1], np.int32), "int32")))
                async def generate():
                    async with client.generate(payload) as events:
                        result = [event async for event in events if event]
                    assert not any(event.get("error") for event in result), result
                    usage = next(event["usage"] for event in result if event.get("usage"))
                    assert usage["prompt_tokens"] == prompt_tokens
                    assert usage["completion_tokens"] > 0
                    return "".join(choice.get("delta", {}).get("content", "")
                        for event in result for choice in event.get("choices", []))

                # Validate the fixture and initialize lazy driver workspaces
                # before measuring the persistent expert footprint.
                expected = await generate()
                await client.set_resident_memory_budget(2 << 30)
                before = await client.status()
                assert before["resident_expert_cache_bytes"] > 48 << 20
                # This is the same allocator/WDDM cap as the environment
                # setting, calibrated to this driver's persistent overhead.
                limit = int(before["resident_memory_used_bytes"]) + (1 << 20)
                assert limit < 2 << 30
                await client.set_resident_memory_budget(limit)
                for _ in range(2):
                    assert await generate() == expected
                    after = await client.status()
                    assert after["resident_memory_budget_bytes"] == limit
                    assert after["resident_memory_used_bytes"] <= limit
                assert after["resident_expert_cache_bytes"] < before["resident_expert_cache_bytes"]
                assert "runtime_expert_workspace_reclaim" in log_path.read_text(errors="replace")
                with pytest.raises(BackendError):
                    await client.set_resident_memory_budget(1)
                assert (await client.status())["resident_memory_budget_bytes"] == limit
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    asyncio.run(run())
