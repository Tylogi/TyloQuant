"""Retained decode graphs must survive reset, prefix restore and eager prefill."""
import asyncio
import os
import re
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import mtp_fixture, qwen_model_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("allocation,kv_bits,drafts", [
    (mode, 0, 3) for mode in ("preallocated", "dynamic", "tiered", "tiered_ssd", "tiered_dynamic_ram")
] + [("preallocated", 0, 5)]
  + [(mode, bits, 5) for mode in ("tiered_ssd", "tiered_dynamic_ram") for bits in (0, 2.5, 4)])
def test_conversation_graph_reuse(tmp_path, allocation, kv_bits, drafts):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = mtp_fixture("qwen", 1)
    for name, value in list(weights.items()):
        if name.endswith(".experts.gate_up.weight"):
            for role, matrix in zip(("gate", "up"), np.split(value, 2, axis=1)):
                weights[name.replace("gate_up", role)] = matrix.copy()
            del weights[name]
    for name, value in list(weights.items()):
        if ".experts." in name:
            experts, output, width = value.shape
            packed = quantize(value.reshape(experts*output, width), NintSpec(4, 8, 6))
            weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
        elif ".ngram.shard." in name:
            weights[name] = quantize(value, NintSpec(4, 8, 6))
    model = tmp_path / "conversation.mfq"
    write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)
    streamed = allocation in {"tiered_ssd", "tiered_dynamic_ram"}

    async def run(reuse):
        log_path = tmp_path / f"worker-{reuse}.log"
        env = dict(os.environ, MFQ_QWEN4_REUSE_GRAPHS=str(int(reuse)),
            MFQ_MTP_CACHE_ONLY_PRIME=str(int(reuse)),
            MFQ_KV_STREAM_GRAPHS=str(int(reuse)),
            MFQ_KV_PLACEMENT="auto_ssd" if streamed else "auto" if allocation == "tiered" else "vram",
            MFQ_KV_ALLOCATION="preallocated" if allocation.startswith("tiered") else allocation,
            MFQ_KV_TURBOQUANT_BITS=str(kv_bits), MFQ_MOE_PIPELINE="1",
            MFQ_MOE_ROUTER_LOOKAHEAD="0", MFQ_MOE_TWO_STAGE_FFN="1",
            MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_PREFILL_LAYER="0",
            MFQ_MOE_PREFILL_GRAPH="0", MFQ_MOE_HOST_CACHE_BYTES="1048576",
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE="0", MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES="67108864",
            MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES="0", MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES="0",
            MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS="16",
            MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / f"prefix-{reuse}"))
        if allocation.startswith("tiered"):
            c = config["text_config"]
            layers = c["layer_types"].count("full_attention") + c["mtp_num_hidden_layers"]
            index_bytes = layers * c["indexer_head_dim"] * (128 + 128 // c["indexer_compress_ratio"]) * 2
            raw_bytes = layers * c["num_key_value_heads"] * c["head_dim"] * 4 * (8 if streamed else 64)
            env.update(MFQ_KV_RAM_PREALLOCATED="0" if allocation == "tiered_dynamic_ram" else "1",
                MFQ_KV_GPU_BUDGET_BYTES=str(index_bytes + raw_bytes),
                MFQ_KV_RAM_LIMIT_BYTES=str(256 if streamed else 1 << 20), MFQ_MEMORY_ARCHITECTURE="discrete")
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "128",
                "--moe-gpu-cache-gb", "0.01", "--yarn-context-extension",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                await client.status()
                initial_log = log_path.read_text(errors="replace")
                if reuse:
                    # Startup warms graphs up to the budgeted initial draft depth;
                    # deeper windows are captured on first use.
                    depth = int(re.search(r"runtime_mtp_flexible_budget .*initial_depth=(\d+)", initial_log)[1])
                    warmup = re.search(r"runtime_qwen4_graph_warmup target_captures=(\d+) "
                                       r"maximum_draft_depth=(\d+) predictor_captures=(\d+)", initial_log)
                    assert warmup and int(warmup[2]) == depth and int(warmup[1]) == depth + 1
                    assert int(warmup[3]) > depth
                    if not streamed:
                        assert "runtime_qwen4_mtp_warmup tokens=64" in initial_log
                    else:
                        assert "segments=2" in initial_log
                captures = initial_log.count("runtime_qwen4_graph_capture")
                outputs = []
                # A different suffix requires eager prefill into graph-owned recurrent
                # storage. The last repeated prompt then exercises prefix restoration.
                prompts = ("ab"*48, "ab"*48+"ba"*12, "ab"*48+"ba"*12, "ab"*48+"ba"*12)
                for index, prompt in enumerate(prompts):
                    if index == 3:
                        # Resizing expert arenas must discard all captured pointers
                        # before the next request restores its prefix.
                        await client.set_resident_memory_budget(1 << 30)
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 12,
                        "enable_mtp": True, "mtp_max_draft_tokens": drafts, "stream": True}) as events:
                        chunks = [event async for event in events if event]
                    outputs.append("".join(choice.get("delta", {}).get("content", "")
                        for event in chunks for choice in event.get("choices", [])))
                    await client.status()
                    if reuse and drafts == 3 and index < 3:
                        assert log_path.read_text(errors="replace").count("runtime_qwen4_graph_capture") == captures
                assert outputs[1] == outputs[2] == outputs[3]
                if allocation == "preallocated" and drafts == 5:
                    for _ in range(2):
                        # Independent short prompts must reuse bounded windows
                        # after a prior request has retained all MTP widths.
                        async with client.generate({"messages": [{"role": "user", "content": "ba"*15+"b"}],
                            "mfq_preformatted_prompt": "ba"*15+"b", "mfq_prefix_cache_enabled": False,
                            "temperature": 0, "max_tokens": 12, "enable_mtp": True,
                            "mtp_max_draft_tokens": drafts, "stream": True}) as events:
                            chunks = [event async for event in events if event]
                        outputs.append("".join(choice.get("delta", {}).get("content", "")
                            for event in chunks for choice in event.get("choices", [])))
                    assert outputs[-1] == outputs[-2]
                status = await client.status()
                if not streamed:
                    assert status["prefix_cache_hit_tokens"] > 0
                else:
                    assert status["qsa_kv_ssd_reads"] > 0
                return outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    expected = asyncio.run(run(False))
    assert asyncio.run(run(True)) == expected


@pytest.mark.parametrize('batched_dma', [False, True])
@pytest.mark.parametrize('partial_pin', [False, True])
def test_kv_expert_loan_large_fields_in_decode_graph(tmp_path, batched_dma, partial_pin):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = qwen_model_fixture(ple=False, hidden=256)
    config['text_config'].update(moe_intermediate_size=1024, shared_expert_intermediate_size=1024, max_position_embeddings=32768, num_experts_per_tok=3)
    for name, value in list(weights.items()):
        if name.endswith('.experts.gate_up.weight'):
            for role, matrix in zip(('gate', 'up'), np.split(value, 2, axis=1)):
                weights[name.replace('gate_up', role)] = np.tile(matrix, (1, 64, 1))
            del weights[name]
        elif name.endswith('.experts.down.weight'):
            weights[name] = np.tile(value, (1, 1, 64)) / 64
        elif name.endswith(('.shared_expert.gate.weight', '.shared_expert.up.weight')):
            weights[name] = np.tile(value, (64, 1))
        elif name.endswith('.shared_expert.down.weight'):
            weights[name] = np.tile(value, (1, 64)) / 64
    for name, value in list(weights.items()):
        if '.experts.' in name:
            experts, output, width = value.shape
            packed = quantize(value.reshape(experts * output, width), NintSpec(4, 8, 6))
            weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
        elif '.shared_expert.' in name and not name.endswith('.router.weight'):
            weights[name] = quantize(value, NintSpec(4, 8, 6))
    model = tmp_path / 'large-fields.mfq'
    write_qwen_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(enabled):
        env = dict(os.environ, MFQ_KV_PLACEMENT='auto', MFQ_KV_ALLOCATION='preallocated', MFQ_KV_RAM_PREALLOCATED='1',
            MFQ_KV_GPU_BUDGET_BYTES=str(5 << 20), MFQ_KV_RAM_LIMIT_BYTES=str(8 << 20), MFQ_KV_EXPERT_LOAN=str(int(enabled)),
            MFQ_KV_TURBOQUANT_BITS='0', MFQ_MOE_PIPELINE='1', MFQ_MOE_TWO_STAGE_FFN='1', MFQ_MOE_PRELOAD_ALL='1',
            MFQ_MOE_RAM_PCIE='1', MFQ_MOE_RAM_PCIE_FRACTION='1', MFQ_MOE_FFN_TRANSFER_PHASES='1', MFQ_MOE_TRANSFER_CACHE='1',
            MFQ_MOE_DMA_GRAPH_BATCH=str(int(batched_dma)),
            MFQ_MOE_PREFILL_LAYER='0', MFQ_MOE_PREFILL_GRAPH='0', MFQ_MOE_HOST_CACHE_BYTES=str(64 << 20),
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE='1', MFQ_MEMORY_ARCHITECTURE='discrete')
        if partial_pin:
            env['MFQ_MOE_RAM_PIN_GIB'] = '0.00025'
        with (tmp_path / f'worker-{enabled}.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(executable, '--model', str(model), '--tokenizer', str(tokenizer),
                '--transport', 'stdio', '--ctx-size', '32768', '--moe-gpu-cache-gb', '0.0005',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=20)
            try:
                await client.status()
                async def generate():
                    async with client.generate({'messages': [{'role': 'user', 'content': 'ab' * 30}],
                        'mfq_preformatted_prompt': 'ab' * 30, 'temperature': 0, 'max_tokens': 24, 'enable_mtp': False, 'stream': True}) as events:
                        chunks = [event async for event in events if event]
                    return ''.join(choice.get('delta', {}).get('content', '') for event in chunks for choice in event.get('choices', []))
                result = await asyncio.wait_for(generate(), 20)
                status = await client.status()
                if enabled:
                    assert status['kv_memory_expert_loan_bytes'] > 0
                    assert status['kv_memory_expert_loan_hit_bytes'] > 0
                return result
            finally:
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
                with suppress(Exception):
                    await client.aclose()
                log.write(f'\nworker_exit={process.returncode}\n'.encode())
    expected = asyncio.run(run(False))
    assert asyncio.run(run(True)) == expected


@pytest.mark.parametrize('mtp', [False, True])
def test_layer_major_prefill_continues_across_bounded_chunks(tmp_path, mtp):
    executable = os.environ.get('MFQ_CUDA_RESOURCE_TEST_RUNTIME')
    if not executable:
        pytest.skip('CUDA runtime required')
    config, weights = mtp_fixture('qwen', 1)
    config['text_config']['max_position_embeddings'] = 32768
    for name, value in list(weights.items()):
        if name.endswith('.experts.gate_up.weight'):
            for role, matrix in zip(('gate', 'up'), np.split(value, 2, axis=1)):
                weights[name.replace('gate_up', role)] = matrix.copy()
            del weights[name]
    for name, value in list(weights.items()):
        if '.experts.' in name:
            experts, output, width = value.shape
            packed = quantize(value.reshape(experts * output, width), NintSpec(4, 8, 6))
            weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
    model = tmp_path / 'bounded-prefill.mfq'
    write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(layer_major):
        env = dict(os.environ, MFQ_QWEN4_REUSE_GRAPHS='1', MFQ_KV_PLACEMENT='auto',
            MFQ_KV_ALLOCATION='preallocated', MFQ_KV_TURBOQUANT_BITS='0', MFQ_KV_RAM_PREALLOCATED='1',
            MFQ_MOE_PIPELINE='1', MFQ_MOE_TWO_STAGE_FFN='1', MFQ_MOE_PRELOAD_ALL='1',
            MFQ_MOE_RAM_PCIE='1', MFQ_MOE_RAM_PCIE_FRACTION='1', MFQ_MOE_HOST_CACHE_BYTES=str(16 << 20),
            MFQ_MOE_PREFILL_LAYER='1', MFQ_MOE_PREFILL_LAYER_MAJOR=str(int(layer_major)),
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE='1', MFQ_MEMORY_ARCHITECTURE='discrete')
        with (tmp_path / f'bounded-{layer_major}.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(executable, '--model', str(model),
                '--tokenizer', str(tokenizer), '--transport', 'stdio', '--ctx-size', '32768',
                '--prefill-chunk-size', '2048', '--moe-gpu-cache-gb', '0.1', '--yarn-context-extension',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=60)
            try:
                await client.status()
                outputs = []
                for prompt in ('ab' * 18000, 'ba' * 4200):
                    async with client.generate({'messages': [{'role': 'user', 'content': prompt}],
                        'mfq_preformatted_prompt': prompt, 'temperature': .7, 'top_k': 20, 'top_p': .8,
                        'seed': 1234, 'max_tokens': 1, 'enable_mtp': mtp, 'mtp_max_draft_tokens': 3,
                        'mfq_prefix_cache_enabled': False, 'stream': True}) as events:
                        chunks = [event async for event in events if event]
                    assert not any('error' in event for event in chunks), chunks
                    outputs.append(''.join(choice.get('delta', {}).get('content', '')
                        for event in chunks for choice in event.get('choices', [])))
                    assert outputs[-1], 'The prefill comparison must produce a sampled token'
                return outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
                log.write(f'\nworker_exit={process.returncode}\n'.encode())

    expected = asyncio.run(run(False))
    assert asyncio.run(run(True)) == expected
