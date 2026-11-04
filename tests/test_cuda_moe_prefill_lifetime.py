"""All-GPU expert prefill must not retain a cell per layer and prompt shape."""

import asyncio
from contextlib import suppress
import json
import os

import numpy as np
import pytest

from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_nemotron_runtime import model_fixture, write_nemotron_fixture
from tests.test_native_nint_ple import _fresh_idle_status, _worker_tokenizer


@pytest.fixture
def resident_moe_model(tmp_path):
    config, weights = model_fixture(("moe",) * 16)
    config["llm_config"].update(max_position_embeddings=8192, moe_intermediate_size=8192)
    for name, value in list(weights.items()):
        if name.endswith(".experts.up.weight"):
            value = np.tile(value, (1, 1024, 1))
        elif name.endswith(".experts.down.weight"):
            value = np.tile(value, (1, 1, 1024)) / 1024
        elif ".latent." in name or ".shared_expert." in name:
            weights[name] = quantize(value, NintSpec(8, 8, 6))
            continue
        else:
            continue
        experts, rows, width = value.shape
        packed = quantize(value.reshape(experts * rows, width), NintSpec(4, 8, 6))
        weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
    model = tmp_path / "resident-moe.mfq"
    write_nemotron_fixture(model, config, weights)
    return model, _worker_tokenizer(tmp_path)


async def run_prompts(tmp_path, fixture, lengths, *, vram_limit="4", max_tokens=4):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("MFQ_CUDA_RESOURCE_TEST_RUNTIME required")
    model, tokenizer = fixture
    env = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
               MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_ASSERT_RESIDENT="1", MFQ_MOE_SSD_EXPERTS="0",
               MFQ_MOE_NVQ_DENSE="0", MFQ_MOE_PREFILL_LAYER="0", MFQ_MOE_PREFILL_GROUPS="0",
               MFQ_MOE_HOST_CACHE_BYTES=str(64 << 20),
               MFQ_MOE_ROUTER_LOOKAHEAD="0", MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
               MFQ_KV_TURBOQUANT_BITS="0", MFQ_CUDA_VRAM_LIMIT_GIB=vram_limit,
               MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.01")
    log_path = tmp_path / "worker.log"
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            executable, "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio",
            "--ctx-size", "8192", "--prefill-chunk-size", "2048", "--moe-gpu-cache-gb", "0.05",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
        client = StdioRuntimeClient(process, control_timeout_seconds=60)
        samples = []
        try:
            await client.status()
            for tokens in lengths:
                payload = dict(messages=[dict(role="user", content="ab")], mfq_preformatted_prompt="ab",
                               mfq_benchmark_prompt_tokens=tokens, mfq_prefix_cache_enabled=False,
                               temperature=0, max_tokens=max_tokens, enable_mtp=False, stream=True,
                               stream_options=dict(include_usage=True))
                async with client.generate(payload) as events:
                    chunks = [event async for event in events if event]
                assert not any(event.get("error") for event in chunks), (chunks, log_path.read_text())
                assert any(event.get("usage", {}).get("prompt_tokens") == tokens for event in chunks), chunks
                status = await _fresh_idle_status(client)
                assert status["ssd_expert_enabled"] == 0
                sample = dict(tokens=tokens, resident_bytes=status["gpu_0_resident_bytes"],
                              captures=status["expert_graph_captures"],
                              prefill_ms=status["last_request"]["prefill_ms"])
                samples.append(sample)
                print(json.dumps(sample), flush=True)
        finally:
            with suppress(Exception):
                await client.aclose()
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 5)
    trace = log_path.read_text(errors="replace")
    assert "ram_experts=0" in trace, trace
    assert "cleanup_error" not in trace, trace
    return samples


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_resident_moe_prefill_does_not_accumulate_prompt_shapes(tmp_path, resident_moe_model, max_tokens):
    # The repeated 40-row request also detects cells retained by earlier requests.
    samples = asyncio.run(run_prompts(tmp_path, resident_moe_model, [40, 90, 700, 1500, 40], max_tokens=max_tokens))
    resident = [sample["resident_bytes"] for sample in samples]
    assert max(resident) - min(resident) < 16 << 20, samples
    assert len({sample["captures"] for sample in samples}) == 1, samples


def test_resident_moe_long_prefill_with_small_vram_limit(tmp_path, resident_moe_model):
    asyncio.run(run_prompts(tmp_path, resident_moe_model, [4096], vram_limit="2"))
