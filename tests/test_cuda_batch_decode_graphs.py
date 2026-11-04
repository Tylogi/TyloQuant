"""Native CUDA row-bucket graphs with recurrent and full-attention request state."""

import asyncio
import json
import os
import subprocess
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from tests.test_cuda_kv_quantization_native import write_qwen35_fixture
from tests.test_native_nint_ple import _fresh_idle_status, _models, _worker_tokenizer


def _recurrent_model(path):
    write_qwen35_fixture(path)
    header, weights = io.load(path)
    config = json.loads(weights[MODEL_CONFIG_ASSET])
    config.update(
        max_position_embeddings=512,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
    rng = np.random.default_rng(6318)
    root = "model.block.0.linear_attention."
    for role, shape in {
        "qkv": (128, 32), "gate": (64, 32), "alpha": (2, 32),
        "beta": (2, 32), "output": (32, 64),
    }.items():
        weights[root + role + ".weight"] = rng.normal(0, .03, shape).astype(np.float32)
    weights[root + "conv.weight"] = rng.normal(0, .03, (128, 1, 4)).astype(np.float32)
    weights[root + "norm.weight"] = np.ones(32, np.float32)
    weights[root + "a"] = -np.ones(2, np.float32)
    weights[root + "dt_bias"] = np.zeros(2, np.float32)
    io.save(path, header, weights)


def _mtp_recurrent_model(path):
    _recurrent_model(path)
    header, weights = io.load(path)
    config = json.loads(weights[MODEL_CONFIG_ASSET])
    config.update(mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
    rng = np.random.default_rng(9167)
    weights["predictor.fusion.weight"] = rng.normal(0, .03, (32, 64)).astype(np.float32)
    for name in ("hidden_norm", "embedding_norm", "output_norm"):
        weights[f"predictor.{name}.weight"] = np.zeros(32, np.float32)
    for name, value in list(weights.items()):
        if name.startswith("model.block.1."):
            weights[name.replace("model.block.1.", "predictor.block.0.")] = value.copy()
    graph = json.loads(weights[MODEL_GRAPH_ASSET])
    graph["canonical_naming"]["component_roots"].append("predictor")
    graph["topology"]["predictor_layers"] = 1
    graph["capabilities"].append("mtp")
    graph["components"].append(dict(kind="predictor", tensor_root="predictor",
                                  implementation="next_token_prediction", policy="optional"))
    weights[MODEL_GRAPH_ASSET] = json.dumps(graph).encode()
    io.save(path, header, weights)

@pytest.mark.parametrize("family,paged,requests", [
    ("qwen35", False, 4), ("qwen35", True, 4), ("qwen35", True, 32),
    ("qwen35_homogeneous", True, 32),
    ("flash_next", False, 4), ("flash_next", False, 32),
    ("flash_next_cached", False, 4), ("flash_next_cached", False, 32),
])
def test_uniform_width_batched_mtp_matches_ordinary_greedy(tmp_path, family, paged, requests):
    metrics = _batched_mtp_check(tmp_path, family, paged, requests)
    lifecycle = "0" if family == "qwen35_homogeneous" else "1"
    assert metrics["paused"] == metrics["cancelled"] == lifecycle
    if family == "qwen35_homogeneous":
        assert int(metrics["tokens"]) == 32 * 48
    assert metrics["grammar"] == "1"
    assert int(metrics["drafted"]) > 0


def _batched_mtp_check(tmp_path, family, paged, requests):
    """Run the diagnostic batched-MTP greedy oracle and return its metrics."""
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    model = tmp_path / "mtp.mfq"
    from tests.test_native_nint_ple import _worker_tokenizer

    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_CONTINUOUS_PAGED_KV=str(int(paged)),
                       MFQ_BATCH_MTP_TEST_TOKENIZER=str(tokenizer), MFQ_BATCH_MTP_CHECK="1",
                       MFQ_CONTINUOUS_BATCH_MTP="1")
    # Let every row speculate unless a test fixes a smaller checkpoint budget.
    environment.setdefault("MFQ_BATCH_MTP_CHECKPOINTS", str(requests))
    if family.startswith("qwen35"):
        _mtp_recurrent_model(model)
        if family == "qwen35_homogeneous":
            environment["MFQ_BATCH_MTP_HOMOGENEOUS"] = "1"
    else:
        from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture

        config, weights = mtp_fixture("qwen", 1, hidden=32)
        config["text_config"]["max_position_embeddings"] = 512
        if family == "flash_next_cached":
            from mfq.formats.mfe import MfePool, MfeTensor
            from mfq.formats.nint import NintSpec
            from mfq.quantize.nint_quant import quantize

            for name, value in list(weights.items()):
                if name.endswith(".experts.gate_up.weight"):
                    for role, matrix in zip(("gate", "up"), np.split(value, 2, axis=1)):
                        weights[name.replace("gate_up", role)] = matrix.copy()
                    del weights[name]
            for name, value in list(weights.items()):
                if ".experts." in name:
                    experts, output, width = value.shape
                    packed = quantize(value.reshape(experts * output, width), NintSpec(4, 8, 6))
                    pool = MfePool(np.arange(experts, dtype=np.int32), packed)
                    weights[name] = MfeTensor(value.shape, (pool,))
                elif ".ngram.shard." in name:
                    weights[name] = quantize(value, NintSpec(4, 8, 6))
            environment.update(MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
                               MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_HOST_CACHE_BYTES="1048576")
        write_qwen_fixture(model, config, weights, True)
    result = subprocess.run(
        [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
         "--continuous-batching-check-requests", str(requests), "--moe-gpu-cache-gb",
         ".01" if family == "flash_next_cached" else "0"],
        env=environment,
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_mtp_check "))
    metrics = dict(field.split("=", 1) for field in line.split()[1:])
    assert metrics["greedy"] == "PASS"
    assert metrics["mtp"] == "1"
    return metrics

def test_flash_next_batched_mtp_drafts_within_fixed_checkpoints(tmp_path, monkeypatch):
    # One checkpoint slot for 32 rows: drafts shrink, memory and outputs do not change.
    monkeypatch.setenv("MFQ_BATCH_MTP_CHECKPOINTS", "1")
    test_uniform_width_batched_mtp_matches_ordinary_greedy(tmp_path, "flash_next", False, 32)

@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
def test_batched_mtp_without_checkpoints_stays_greedy(tmp_path, monkeypatch, family):
    # A zero budget keeps batched MTP enabled but never speculates.
    monkeypatch.setenv("MFQ_BATCH_MTP_CHECKPOINTS", "0")
    metrics = _batched_mtp_check(tmp_path, family, False, 4)
    assert int(metrics["drafted"]) == 0

@pytest.mark.parametrize("opt_in", [True, False])
@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
def test_mtp_capability_and_seeded_stochastic_exclusion(tmp_path, family, opt_in):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("MFQ_CUDA_RESOURCE_TEST_RUNTIME must point to CUDA mfq-runtime")
    from mfq.server.runtime.client import StdioRuntimeClient
    from tests.test_native_nint_ple import _worker_tokenizer
    from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture

    model = tmp_path / "mtp.mfq"
    if family == "qwen35":
        _mtp_recurrent_model(model)
    else:
        config, weights = mtp_fixture("qwen", 1, hidden=32)
        config["text_config"]["max_position_embeddings"] = 512
        write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run():
        with (tmp_path / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                executable, "--model", str(model), "--tokenizer", str(tokenizer),
                "--transport", "stdio", "--ctx-size", "512", "--continuous-batching", "2",
                "--moe-gpu-cache-gb", "0", stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log,
                env=dict(os.environ, MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
                         MFQ_CONTINUOUS_BATCH_MTP=str(int(opt_in))))
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                status = await client.status()
                assert status["continuous_batching_mtp_supported"] == int(opt_in)
                assert status["continuous_batching_mtp_stochastic_supported"] == 0
                assert status["continuous_batching_mtp_graph_supported"] == 0

                async def generate(stochastic, mtp=True):
                    params = dict(messages=[dict(role="user", content="ab" * 10)],
                                  mfq_preformatted_prompt="ab" * 10, temperature=.7 if stochastic else 0,
                                  top_k=4 if stochastic else 1, top_p=.9, seed=731, max_tokens=12,
                                  enable_mtp=mtp, stream=True)
                    async with client.generate(params) as events:
                        chunks = [event async for event in events if event]
                    return "".join(choice.get("delta", {}).get("content", "")
                                   for event in chunks for choice in event.get("choices", []))

                assert await generate(True) == await generate(True)
                await asyncio.sleep(.3)
                assert (await client.status()).get("continuous_batching_admissions", 0) == 0
                expected = await generate(False, False)
                assert await asyncio.gather(generate(False), generate(False)) == [expected, expected]
                if not opt_in:
                    # Without the opt-in, greedy MTP requests stay exclusive.
                    assert (await client.status()).get("continuous_batching_mtp_batches", 0) == 0
                    return
                for _ in range(30):
                    status = await client.status()
                    if status.get("continuous_batching_mtp_batches", 0) > 0:
                        break
                    await asyncio.sleep(.02)
                assert status["continuous_batching_mtp_batches"] > 0
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    try:
        asyncio.run(run())
    except Exception as error:
        raise AssertionError((tmp_path / "worker.log").read_text(errors="replace")) from error

@pytest.mark.parametrize("family", ["qwen35", "flash_next"])
def test_batched_mtp_all_accepted_windows_commit(tmp_path, family):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    model = tmp_path / "accept.mfq"
    if family == "qwen35":
        _mtp_recurrent_model(model)
        header, weights = io.load(model)
        weights["model.output.weight"][:] = 0
        io.save(model, header, weights)
    else:
        from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture

        config, weights = mtp_fixture("qwen", 1, hidden=32)
        config["text_config"]["max_position_embeddings"] = 512
        weights["model.output.weight"][:] = 0
        write_qwen_fixture(model, config, weights, True)
    result = subprocess.run(
        [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
         "--continuous-batching-check-requests", "4", "--moe-gpu-cache-gb", "0"],
        env=dict(os.environ, MFQ_BATCH_MTP_CHECK="1", MFQ_CONTINUOUS_BATCH_MTP="1"),
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_mtp_check "))
    metrics = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(metrics["drafted"]) > 0
    assert metrics["accepted"] == metrics["drafted"]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("requests", [2, 32])
def test_qwen_continuous_graph_reuses_bucket_and_preserves_inactive_state(tmp_path, paged, requests):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    model = tmp_path / "recurrent.mfq"
    _recurrent_model(model)
    result = subprocess.run(
        [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
         "--continuous-batching-check-requests", str(requests)],
        env=dict(os.environ, MFQ_CONTINUOUS_PAGED_KV=str(int(paged))),
        text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_membership_check "))
    metrics = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(metrics["captures"]) == 1
    assert int(metrics["bucket_reuses"]) > 0
    assert metrics["serial"] == metrics["inactive_state_bits"] == "PASS"
    assert metrics["padding_routes"] == "PASS"
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_sampling_membership_check "))
    sampling = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(sampling["captures"]) == 1
    assert int(sampling["bucket_reuses"]) > 0
    assert sampling["serial"] == sampling["inactive_state_bits"] == sampling["inactive_rng"] == "PASS"
    assert sampling["padding_routes"] == "PASS"
    assert "continuous_batching_check PASS" in result.stdout
    if requests == 32:
        assert "continuous_batching_growth_check requests=32 boundary=256 PASS" in result.stdout


@pytest.mark.parametrize("requests", [2, 8, 16, 32])
@pytest.mark.parametrize("kv_bits", ["0", "3.5"])
def test_qsa_independent_graph_heterogeneous_requests_match_serial(tmp_path, requests, kv_bits):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512)
    result = subprocess.run(
        [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
         "--continuous-batching-check-requests", str(requests), "--moe-gpu-cache-gb", "0"],
        env=dict(os.environ, MFQ_QSA_INDEPENDENT_BATCH="1", MFQ_KV_TURBOQUANT_BITS=kv_bits),
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "continuous_batching_check PASS" in result.stdout
    membership = next(line for line in result.stdout.splitlines()
                      if line.startswith("continuous_batching_membership_check "))
    metrics = dict(field.split("=", 1) for field in membership.split()[1:])
    assert int(metrics["captures"]) == 1
    assert int(metrics["bucket_reuses"]) > 0
    assert metrics["serial"] == metrics["inactive_state_bits"] == metrics["padding_routes"] == "PASS"
    if requests > 2:
        assert f"continuous_batching_growth_check requests={requests} boundary=256 PASS" in result.stdout


@pytest.mark.parametrize("family,requests", [("qwen35", 2), ("flash_next", 2), ("flash_next", 32)])
def test_stochastic_sampling_reuses_graph_bucket_during_membership_changes(tmp_path, family, requests):
    executable = os.environ.get("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS")
    if not executable:
        pytest.skip("MFQ_CUDA_BATCH_GRAPH_DIAGNOSTICS must point to CUDA mfq-diagnostics")
    environment = dict(os.environ)
    if family == "qwen35":
        model = tmp_path / "recurrent.mfq"
        _recurrent_model(model)
    else:
        from tests.test_native_nint_ple import _models

        _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512, quantized_experts=requests == 2)
        if requests == 2:
            environment.update(MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1", MFQ_MOE_PRELOAD_ALL="1",
                               MFQ_MOE_ASSERT_RESIDENT="1", MFQ_MOE_HOST_CACHE_BYTES="8388608")
    arguments = [executable, "--model", str(model), "--ctx-size", "512", "--check-continuous-batching",
                 "--continuous-batching-check-requests", str(requests)]
    if family == "flash_next" and requests == 32:
        arguments += ["--moe-gpu-cache-gb", "0"]
    result = subprocess.run(
        arguments,
        env=environment,
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("continuous_batching_sampling_membership_check "))
    metrics = dict(field.split("=", 1) for field in line.split()[1:])
    assert int(metrics["captures"]) == 1
    assert int(metrics["bucket_reuses"]) == int(metrics["steps"]) - 1
    assert all(metrics[name] == "PASS" for name in ("serial", "inactive_state_bits", "inactive_rng", "padding_routes"))
    assert "continuous_batching_sampling_oracle logits=tied" in result.stdout


@pytest.mark.parametrize("budget", [32, 2048])
@pytest.mark.parametrize("cells", [16, 496])
@pytest.mark.parametrize("kv_bits", ["0", "3.5"])
def test_flash_next_bucket_graph_uses_physical_kv_tiers(tmp_path, budget, cells, kv_bits):
    from mfq.server.runtime.client import StdioRuntimeClient

    executable = os.environ.get("MFQ_FLASH_NEXT_CUDA_RUNTIME")
    if not executable:
        pytest.skip("MFQ_FLASH_NEXT_CUDA_RUNTIME must point to CUDA mfq-runtime")
    _, model = _models(tmp_path, qsa=True, ple_layer=3, context_size=512, quantized_experts=True)
    header, weights = io.load(model)
    config = json.loads(weights[MODEL_CONFIG_ASSET])
    config["text_config"].update(num_key_value_heads=2, indexer_budget=budget)
    for role in ("key", "value"):
        name = f"model.block.1.attention.{role}.weight"
        weights[name] = np.tile(weights[name], (2, 1))
    weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
    io.save(model, header, weights)
    tokenizer = _worker_tokenizer(tmp_path)
    # Packed rows include one norm word per head: K uses 3 bits and V uses 4.
    cell_bytes = 512 if kv_bits == "0" else 2 * ((1 + 6) + (1 + 8)) * 4
    environment = dict(os.environ, MFQ_MOE_PIPELINE="1", MFQ_MOE_TWO_STAGE_FFN="1",
                       MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_HOST_CACHE_BYTES="8388608",
                       MFQ_MEMORY_ARCHITECTURE="discrete", MFQ_KV_RAM_PREALLOCATED="1",
                       MFQ_QSA_INDEPENDENT_BATCH="1", MFQ_KV_TURBOQUANT_BITS=kv_bits,
                       MFQ_KV_GPU_BUDGET_BYTES=str(98304 + cells * cell_bytes))
    # A small GPU tier exercises eviction and cold reads; a small RAM tier
    # exposes writes through the logical [heads, 512, width] view immediately.
    prompts = ["ab" * count for count in (12, 25, 40)]

    async def run(capacity):
        log_path = tmp_path / f"tiered-{capacity}.log"
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                executable, "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio",
                "--ctx-size", "512", "--continuous-batching", str(capacity), "--moe-gpu-cache-gb", "0.00004",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                env=dict(environment, MFQ_CONTINUOUS_BATCH_CUDA_GRAPH=str(int(capacity > 1))),
            )
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                await client.status()

                async def generate(prompt):
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                            "mfq_preformatted_prompt": prompt, "temperature": 0,
                            "top_k": 1, "max_tokens": 32, "enable_mtp": False, "stream": True}) as events:
                        output = [event async for event in events if event is not None]
                    return "".join(choice.get("delta", {}).get("content", "")
                                   for event in output for choice in event.get("choices", []))

                single = await generate(prompts[0])
                outputs = (await asyncio.gather(*(generate(prompt) for prompt in prompts)) if capacity > 1
                           else [await generate(prompt) for prompt in prompts])
                status = await _fresh_idle_status(client)
                assert status["kv_memory_resident_cells"] == cells
                assert status["kv_memory_ram_bytes"] > 0
                if capacity > 1:
                    assert status["continuous_batching_cuda_graph_captures"] > 0
                    assert status["continuous_batching_cuda_graph_bucket_reuses"] > 0
                assert "exclusive=1" in log_path.read_text(errors="replace")
                return single, outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await process.wait()

    async def check():
        reference = await run(1)
        assert reference[0] and all(reference[1])
        assert await run(4) == reference

    asyncio.run(check())
