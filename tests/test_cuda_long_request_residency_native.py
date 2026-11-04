"""Repeated 32k prompts must stay within the native cache and arena bounds."""

import asyncio
from contextlib import suppress
import json
import os

import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_nemotron_runtime import model_fixture, write_nemotron_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("prefix_cache,vram_gib", [(True, 6), (True, 2), (False, 6)])
def test_repeated_32k_request_has_bounded_residency(tmp_path, prefix_cache, vram_gib):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = model_fixture()
    config["llm_config"]["max_position_embeddings"] = 32768
    model = tmp_path / "long-request.mfq"
    write_nemotron_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    hot_limit = 64 << 20
    log_path = tmp_path / "worker.log"

    async def run():
        environment = dict(os.environ,
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE="0" if prefix_cache else "1",
            MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES=str(hot_limit),
            MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES="0", MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES="0",
            MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / "prefix"), MFQ_KV_PLACEMENT="vram",
            MFQ_KV_ALLOCATION="dynamic", MFQ_KV_TURBOQUANT_BITS="0", MFQ_CUDA_VRAM_LIMIT_GIB=str(vram_gib),
            MFQ_RUNTIME_MAX_KV_SESSIONS="4", MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION="4")
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "32768",
                "--prefill-chunk-size", "2048", "--moe-gpu-cache-gb", "0.01",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
            client = StdioRuntimeClient(process, control_timeout_seconds=60)
            try:
                await client.status()

                async def generate(tokens):
                    prompt = "a" * tokens
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt,
                        "temperature": 0, "max_tokens": 16, "enable_mtp": False, "stream": True,
                        "stream_options": {"include_usage": True}}) as events:
                        result = [event async for event in events if event]
                    assert not any(event.get("error") for event in result), result
                    assert any(event.get("usage", {}).get("prompt_tokens") == tokens for event in result), result
                    # Scheduler status snapshots refresh every 250 ms.
                    await asyncio.sleep(0.3)
                    status = await client.status()
                    keys = ("gpu_0_resident_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes",
                        "gpu_0_kv_bytes", "kv_cache_bytes", "gpu_0_prefix_cache_bytes",
                        "prefix_cache_hot_bytes", "prefix_cache_hot_blocks", "prefix_cache_hits",
                        "prefix_cache_resident_sessions")
                    row = {key: status[key] for key in keys if key in status}
                    assert "cuda_allocated_bytes" in row, status
                    print(json.dumps({"prefix_cache": prefix_cache, "vram_gib": vram_gib,
                        "prompt_tokens": tokens, **row}), flush=True)
                    return row

                await generate(32)
                samples = [await generate(32582) for _ in range(8)]
                for row in samples:
                    assert row.get("prefix_cache_hot_bytes", 0) <= hot_limit
                    assert row.get("prefix_cache_hot_blocks", 0) <= 4
                # Anonymous snapshots are limited to four cache owners. Once
                # those slots have warmed, identical prompts cannot add storage.
                for key in ("cuda_allocated_bytes", "cuda_reserved_bytes", "gpu_0_resident_bytes"):
                    ceiling = max(row[key] for row in samples[:4])
                    assert all(row[key] <= ceiling for row in samples[4:]), (key, samples)
                if prefix_cache:
                    assert samples[-1]["prefix_cache_hot_bytes"] > 0
                    # Under the 2 GiB cap prefill may reclaim the snapshot before
                    # it is reused; only the roomier cap must hit.
                    if vram_gib == 6:
                        assert samples[-1]["prefix_cache_hits"] > samples[0]["prefix_cache_hits"]
                else:
                    assert all(row.get("prefix_cache_hot_bytes", 0) == 0 for row in samples)
                await client.clear_cache()
                await asyncio.sleep(0.3)
                status = await client.status()
                assert status.get("prefix_cache_hot_bytes", 0) == 0
                print(json.dumps({"prefix_cache": prefix_cache, "vram_gib": vram_gib, "after_clear": {
                    key: status.get(key) for key in samples[-1]}}), flush=True)
                # Reapply the existing budget to exercise idle pool trimming
                # without lowering the request's context or memory allowance.
                await client.set_resident_memory_budget(vram_gib << 30)
                await asyncio.sleep(0.3)
                reclaimed = await client.status()
                assert reclaimed["gpu_0_resident_bytes"] <= samples[-1]["gpu_0_resident_bytes"]
                print(json.dumps({"prefix_cache": prefix_cache, "vram_gib": vram_gib, "after_reclaim": {
                    key: reclaimed.get(key) for key in samples[-1]}}), flush=True)
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
        assert "cleanup_error" not in log_path.read_text(errors="replace")

    asyncio.run(asyncio.wait_for(run(), timeout=300))
