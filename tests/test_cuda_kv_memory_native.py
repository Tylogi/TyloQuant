"""Native placement policies: capacity admission, host access and spill ordering."""
import asyncio
import os
from contextlib import suppress
from pathlib import Path

import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import qwen_model_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("placement,allocation,gpu_limit,ram_limit", [
    ("vram", "preallocated", 0, 0),
    ("ram", "preallocated", 0, 1 << 20),
    ("auto", "preallocated", 8192, 1 << 20),
    ("auto_ssd", "preallocated", 8192, 4096),
    ("auto", "dynamic", 0, 0),
])
@pytest.mark.parametrize("automatic_ram", [False, True])
def test_kv_policy_generation(tmp_path, placement, allocation, gpu_limit, ram_limit, automatic_ram):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = qwen_model_fixture(ple=False)
    model = tmp_path / "tiny.mfq"
    write_qwen_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(mode, limit, ram, name):
        env = dict(os.environ, MFQ_KV_PLACEMENT=mode, MFQ_KV_ALLOCATION=allocation,
            MFQ_KV_GPU_LIMIT_BYTES=str(limit), MFQ_KV_RAM_LIMIT_BYTES=str(ram),
            MFQ_KV_TURBOQUANT_BITS="0", MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
            MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / name))
        if automatic_ram:
            env.update(MFQ_MEMORY_ARCHITECTURE="discrete", MFQ_KV_AUTO_STREAM_MIN_CONTEXT="64", MFQ_KV_RESIDENT_CELLS="32")
        with (tmp_path / (name + ".log")).open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "128",
                "--yarn-context-extension", stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                before = await client.status()
                assert before["kv_memory_supported"] == 1
                assert before["kv_memory_dynamic"] == (allocation == "dynamic")
                assert before.get("qsa_kv_ssd_bytes", 0) == 0
                if mode == "ram":
                    assert before["kv_memory_ram_bytes"] > 0
                if automatic_ram and mode == "auto" and allocation == "preallocated" and not limit:
                    assert before["kv_memory_auto_ram"] == 1
                    assert before["kv_memory_ram_bytes"] > 0
                results = []
                for count in (90, 100, 100):
                    prompt = "ab" * count
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 4,
                        "enable_mtp": False, "stream": True}) as events:
                        chunks = [event async for event in events if event]
                    results.append("".join(choice.get("delta", {}).get("content", "")
                        for event in chunks for choice in event.get("choices", [])))
                after = await client.status()
                assert results[1] == results[2]
                if limit and not after.get("kv_memory_auto_ram"):
                    assert after["qsa_kv_gpu_evictions"] > 0
                    assert after["qsa_kv_ram_hits"] > 0
                    assert bool(after["qsa_kv_ssd_enabled"]) == (mode == "auto_ssd")
                    # Request cleanup releases live blocks; writes record completed SSD use.
                    deadline = asyncio.get_running_loop().time() + 5
                    while (
                        (after["qsa_kv_pending_bytes"] or after["qsa_kv_ssd_bytes"])
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        await asyncio.sleep(0.01)
                        after = await client.status()
                    assert after["qsa_kv_pending_bytes"] == 0
                    assert bool(after["qsa_kv_ssd_written_bytes"]) == (mode == "auto_ssd")
                    assert after["qsa_kv_ssd_bytes"] == 0
                return results
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    expected = asyncio.run(run("vram", 0, 0, "baseline"))
    assert asyncio.run(run(placement, gpu_limit, ram_limit, "policy")) == expected


@pytest.mark.parametrize("placement,budget,ram,preallocated", [
    ("auto", None, 1 << 20, True),
    ("auto", 12288, 1 << 20, True),
    ("auto", 20480, 1 << 20, True),
    ("auto_ssd", 20480, 1 << 20, True),
    ("auto_ssd", 20480, 4096, True),
    ("auto_ssd", 12288, 0, True),
    ("auto_ssd", 20480, 4096, False),
    ("auto_ssd", 12288, 0, False),
    ("auto_ssd", None, 1 << 20, False),
    ("vram", None, 0, True),
    ("ram", None, 1 << 20, True),
])
def test_budgeted_kv_tiers_generate_and_restore(tmp_path, placement, budget, ram, preallocated):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = qwen_model_fixture(ple=False)
    model = tmp_path / "tiny.mfq"
    write_qwen_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(baseline):
        env = dict(os.environ, MFQ_KV_PLACEMENT="vram" if baseline else placement,
            MFQ_KV_ALLOCATION="preallocated", MFQ_KV_RAM_PREALLOCATED="1" if baseline or preallocated else "0",
            MFQ_KV_RAM_LIMIT_BYTES=str(1 << 20 if baseline else ram), MFQ_MEMORY_ARCHITECTURE="discrete",
            MFQ_KV_TURBOQUANT_BITS="0", MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / "ssd"))
        env.pop("MFQ_KV_GPU_BUDGET_BYTES", None)
        if not baseline and budget is not None:
            env["MFQ_KV_GPU_BUDGET_BYTES"] = str(budget)
        name = "baseline" if baseline else "tiered"
        with (tmp_path / (name + ".log")).open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "128", "--yarn-context-extension",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                before = await client.status()
                assert before.get("qsa_kv_ssd_bytes", 0) == 0
                if not baseline and placement == "auto" and budget == 20480:
                    assert before["kv_memory_ram_bytes"] == 8192
                results = []
                for count in (30, 90, 100, 100):
                    prompt = "ab" * count
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 4,
                        "enable_mtp": False, "stream": True}) as events:
                        chunks = [event async for event in events if event]
                    results.append("".join(choice.get("delta", {}).get("content", "")
                        for event in chunks for choice in event.get("choices", [])))
                return results
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    assert asyncio.run(run(False)) == asyncio.run(run(True))
