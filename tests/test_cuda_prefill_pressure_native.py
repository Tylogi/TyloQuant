"""Synthetic prefill replays must preserve state and leave the worker usable."""

import asyncio
import os
from contextlib import suppress

import pytest

from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from tests.test_cuda_nemotron_runtime import MTP_COMPONENT, model_fixture, write_nemotron_fixture
from tests.test_native_nint_ple import _fresh_idle_status, _worker_tokenizer


def _runtime():
    value = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not value:
        pytest.skip("MFQ_CUDA_RESOURCE_TEST_RUNTIME must point to Windows mfq-runtime")
    return value


async def _exercise(model, tokenizer, log_path, *, limit="2", fault=0, failures=1,
                    mtp=False, batch=0, prompt_tokens=768, fault_scope=1, other_tokens=751, cache=False,
                    graphs=False, moe_cache="0", draft_tokens=3):
    env = dict(os.environ, MFQ_CUDA_VRAM_LIMIT_GIB=limit,
               MFQ_TEST_CUDA_PREFILL_FAIL_AFTER=str(fault),
               MFQ_TEST_CUDA_PREFILL_FAIL_COUNT=str(failures),
               MFQ_TEST_CUDA_PREFILL_FAIL_SCOPE=str(fault_scope),
               MFQ_RUNTIME_TRACE_CUDA_GRAPH="1", MFQ_RUNTIME_DISABLE_PREFIX_CACHE=str(int(not cache)),
               MFQ_KV_TURBOQUANT_BITS="0", MFQ_QWEN4_WARMUP_GRAPHS=str(int(graphs)),
               MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.25",
               MFQ_QWEN4_WARMUP_MTP="0", MFQ_QWEN4_REUSE_GRAPHS=str(int(graphs)),
               MFQ_MOE_PIPELINE="1", MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_PREFILL_LAYER="1",
               MFQ_MOE_HOST_CACHE_BYTES="10485760", MFQ_MOE_RAM_PCIE="1", MFQ_MOE_RAM_PCIE_FRACTION="1",
               MFQ_CONTINUOUS_BATCH_MTP="1",
               MFQ_RUNTIME_PREFIX_CACHE_DIR=str(log_path.parent / (log_path.stem + "-prefix")),
               MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES="1048576", MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES="1048576")
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            _runtime(), "--model", str(model), "--tokenizer", str(tokenizer),
            "--transport", "stdio", "--ctx-size", "2048", "--prefill-chunk-size", "128",
            "--continuous-batching", str(batch), "--moe-gpu-cache-gb", moe_cache,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
        client = StdioRuntimeClient(process, control_timeout_seconds=60)
        try:
            await client.status()

            async def generate(tokens, use_mtp=mtp):
                params = dict(messages=[dict(role="user", content="ab")],
                              mfq_preformatted_prompt="ab", mfq_benchmark_prompt_tokens=tokens,
                              mfq_prefix_cache_enabled=cache, temperature=0, max_tokens=8,
                              enable_mtp=use_mtp, mtp_max_draft_tokens=draft_tokens,
                              stream=True, stream_options=dict(include_usage=True))
                if cache:
                    params.pop("mfq_benchmark_prompt_tokens")
                    params["mfq_preformatted_prompt"] = "ab" * tokens
                async with client.generate(params) as events:
                    chunks = [event async for event in events if event]
                assert not any(event.get("error") for event in chunks), chunks
                usage = next(event["usage"] for event in reversed(chunks) if "usage" in event)
                assert usage["prompt_tokens"] == tokens
                assert usage["completion_tokens"] > 0
                metrics = next(event["mfq_metrics"] for event in chunks if "mfq_metrics" in event)
                assert bool(metrics.get("mtp_used", False)) == bool(use_mtp)
                return "".join(choice.get("delta", {}).get("content", "")
                               for event in chunks for choice in event.get("choices", []))

            error = None
            try:
                if batch:
                    long = await asyncio.gather(generate(other_tokens), generate(prompt_tokens), return_exceptions=True)
                    error = next((value for value in long if isinstance(value, BackendError)), None)
                else:
                    long = await generate(prompt_tokens)
            except BackendError as failure:
                error = failure
                long = None
            short = await generate(12, False)
            before = await _fresh_idle_status(client)
            assert await generate(12, False) == short
            after = await _fresh_idle_status(client)
            assert after["cuda_allocated_bytes"] <= before["cuda_allocated_bytes"] + (1 << 20)
            assert after["cuda_reserved_bytes"] <= before["cuda_reserved_bytes"] + (32 << 20)
            return long, short, error, after["cuda_allocated_bytes"]
        finally:
            with suppress(Exception):
                await client.aclose()
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 5)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_prefill_pressure_replays_mamba_attention_and_mtp(tmp_path, mtp, cache):
    _runtime()
    config, weights = model_fixture(mtp=True)
    config["llm_config"]["max_position_embeddings"] = 2048
    model = tmp_path / "pressure.mfq"
    write_nemotron_fixture(model, config, weights, extra_components=(MTP_COMPONENT,))
    tokenizer = _worker_tokenizer(tmp_path)
    expected = asyncio.run(_exercise(model, tokenizer, tmp_path / "baseline.log", mtp=mtp, cache=cache))
    assert expected[2] is None
    actual = asyncio.run(_exercise(model, tokenizer, tmp_path / "pressure.log",
                                  limit="1", fault=80, fault_scope=2, mtp=mtp, cache=cache))
    trace = (tmp_path / "pressure.log").read_text(errors="replace")
    assert actual[:2] == expected[:2], trace
    assert actual[2] is None, trace
    assert actual[3] <= expected[3] + (1 << 20), trace
    assert trace.count("runtime_prefill_retry scope=request") == 1, trace
    if not mtp:
        assert "runtime_prefill_chunk offset=0 tokens=128" in trace.split("runtime_prefill_retry")[0], trace
    assert "injected CUDA prefill allocation failure" in trace
    assert "cleanup_error" not in trace and "capture is active" not in trace


def test_minimum_prefill_failure_does_not_poison_next_request(tmp_path):
    _runtime()
    config, weights = model_fixture()
    config["llm_config"]["max_position_embeddings"] = 2048
    model = tmp_path / "minimum.mfq"
    write_nemotron_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    expected = asyncio.run(_exercise(model, tokenizer, tmp_path / "baseline.log"))
    # 128, 64, 32, 16, 8, 4, 2, 1: fail exactly the complete retry ladder.
    actual = asyncio.run(_exercise(model, tokenizer, tmp_path / "minimum.log", limit="1", fault=1, failures=8))
    assert actual[0] is None
    assert actual[1] == expected[1]
    assert actual[2] is not None and "minimum chunk=1" in str(actual[2])
    assert actual[3] <= expected[3] + (1 << 20)


def _batch_model(tmp_path, family):
    model = tmp_path / "batch.mfq"
    if family == "qwen35":
        from tests.test_cuda_batch_decode_graphs import _mtp_recurrent_model
        from mfq.formats import io
        from mfq.formats.assets import MODEL_CONFIG_ASSET
        import json

        _mtp_recurrent_model(model)
        header, weights = io.load(model)
        config = json.loads(weights[MODEL_CONFIG_ASSET])
        config["max_position_embeddings"] = 2048
        weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
        io.save(model, header, weights)
    else:
        from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture

        config, weights = mtp_fixture("qwen", 1, hidden=32)
        config["text_config"]["max_position_embeddings"] = 2048
        write_qwen_fixture(model, config, weights, True)
    return model


@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
@pytest.mark.parametrize("other_tokens", [12, 751])
def test_batched_prefill_retry_preserves_other_rows(tmp_path, family, other_tokens, _mtp=False):
    _runtime()
    model = _batch_model(tmp_path, family)
    tokenizer = _worker_tokenizer(tmp_path)
    options = dict(batch=2, other_tokens=other_tokens, mtp=_mtp)
    expected = asyncio.run(_exercise(model, tokenizer, tmp_path / "baseline.log", **options))
    actual = asyncio.run(_exercise(model, tokenizer, tmp_path / "retry.log",
                                  fault=40, fault_scope=2, **options))
    trace = (tmp_path / "retry.log").read_text(errors="replace")
    assert actual[:2] == expected[:2], trace
    assert expected[2] is actual[2] is None, trace
    assert "runtime_prefill_retry scope=row" in trace, trace
    assert actual[3] <= expected[3] + (1 << 20), trace
    assert "cleanup_error" not in trace


@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
def test_batched_mtp_prefill_retry_preserves_other_rows(tmp_path, family):
    test_batched_prefill_retry_preserves_other_rows(tmp_path, family, 12, _mtp=True)


@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
def test_minimum_prefill_failure_isolated_from_decoding_row(tmp_path, family):
    _runtime()
    model = _batch_model(tmp_path, family)
    tokenizer = _worker_tokenizer(tmp_path)
    options = dict(batch=2, other_tokens=12)
    expected = asyncio.run(_exercise(model, tokenizer, tmp_path / "baseline.log", **options))
    actual = asyncio.run(_exercise(model, tokenizer, tmp_path / "retry.log",
                                  fault=1, fault_scope=2, failures=8, **options))
    assert actual[0][0] == expected[0][0]
    assert isinstance(actual[0][1], BackendError) and "minimum chunk=1" in str(actual[0][1])
    assert actual[1] == expected[1]
    assert actual[3] <= expected[3] + (1 << 20)


def test_prefill_graph_preparation_failure_recovers_without_changing_mtp(tmp_path):
    from tests.test_native_nint_ple import _models

    _runtime()
    model = _models(tmp_path, mtp=True, qsa=True, context_size=2048, quantized_experts=True)[1]
    tokenizer = _worker_tokenizer(tmp_path)
    options = dict(mtp=True, graphs=True, moe_cache="0.0009", draft_tokens=5)
    expected = asyncio.run(_exercise(model, tokenizer, tmp_path / "baseline.log", **options))
    actual = asyncio.run(_exercise(model, tokenizer, tmp_path / "retry.log", fault=1, **options))
    trace = (tmp_path / "retry.log").read_text(errors="replace")
    assert actual[:2] == expected[:2], trace
    assert expected[2] is actual[2] is None, trace
    assert "runtime_prefill_retry scope=graph" in trace, trace
    assert actual[3] <= expected[3] + (1 << 20), trace
    assert "cleanup_error" not in trace


def test_quantized_moe_prefill_retries_and_releases_activation_workspace(tmp_path):
    from tests.test_native_nint_ple import _models

    _runtime()
    model = _models(tmp_path, mtp=True, qsa=True, context_size=2048, quantized_experts=True)[1]
    tokenizer = _worker_tokenizer(tmp_path)
    options = dict(mtp=True, moe_cache="0.0009")
    expected = asyncio.run(_exercise(model, tokenizer, tmp_path / "baseline.log", **options))
    actual = asyncio.run(_exercise(model, tokenizer, tmp_path / "retry.log", fault=8, **options))
    trace = (tmp_path / "retry.log").read_text(errors="replace")
    assert actual[:2] == expected[:2], trace
    assert expected[2] is actual[2] is None, trace
    assert "runtime_prefill_retry scope=request chunk=64 " in trace, trace
    assert "moe_row_bytes=48" in trace, trace
    assert actual[3] <= expected[3] + (1 << 20), trace
    assert "cleanup_error" not in trace
