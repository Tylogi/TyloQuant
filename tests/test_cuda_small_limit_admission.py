"""Small-limit stdio admission probes, using generated checkpoints only."""

import asyncio
import json
import os
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from mfq.formats.header import FileHeader
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_batch_decode_graphs import _recurrent_model
from tests.test_native_nint_ple import _fresh_idle_status, _models, _worker_tokenizer


def _glm_dsa_model(path):
    from mfq.formats.mfe import MfePool, MfeTensor
    from mfq.formats.nint import NintSpec
    from mfq.quantize.nint_quant import quantize

    rng = np.random.default_rng(5505)
    hidden, heads, latent, query_rank = 32, 64, 512, 16
    config = dict(
        model_type="glm_dsa_text", vocab_size=32, hidden_size=hidden, intermediate_size=32,
        num_hidden_layers=2, max_position_embeddings=8192, num_attention_heads=heads,
        num_key_value_heads=heads, head_dim=256, rms_norm_eps=1e-5,
        layer_types=["glm_dsa"] * 2, indexer_types=["full", "shared"], mlp_layer_types=["sparse", "dense"],
        q_lora_rank=query_rank, kv_lora_rank=latent, qk_head_dim=256,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        index_head_dim=128, index_n_heads=32, index_topk=2048,
        attention_bias=False, rope_interleave=True, tie_word_embeddings=False, hidden_act="silu",
        num_experts=3, num_experts_per_tok=2, n_group=1, topk_group=1,
        n_shared_experts=1, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid", topk_method="noaux_tc", norm_topk_prob=True, eos_token_id=31,
    )

    def weight(shape):
        return rng.normal(0, .05, shape).astype(np.float32)

    def projection(shape):
        return quantize(weight(shape), NintSpec(4, 24, 6))

    def headwise(shape):
        values = weight(shape)
        packed = quantize(values.reshape(shape[0] * shape[1], shape[2]), NintSpec(4, 24, 6))
        return MfeTensor(shape, (MfePool(np.arange(shape[0], dtype=np.int32), packed),))

    tensors = {"model.token_embedding.weight": weight((32, hidden)).astype(np.float16),
               "model.output.weight": np.zeros((32, hidden), np.float32),
               "model.output_norm.weight": np.zeros(hidden, np.float32)}
    # Reuse encoded synthetic headwise banks between layers.
    query_embedding = headwise((heads, latent, 192))
    output_unembedding = headwise((heads, 256, latent))
    for layer in range(2):
        root = f"model.block.{layer}."
        attention = root + "attention."
        tensors[attention + "norm.weight"] = np.zeros(hidden, np.float32)
        tensors[root + "mlp.norm.weight"] = np.zeros(hidden, np.float32)
        tensors[attention + "query_a_norm.weight"] = np.zeros(query_rank, np.float32)
        tensors[attention + "key_value_a_norm.weight"] = np.zeros(latent, np.float32)
        for role, shape in {"query_a": (query_rank, hidden), "key_value_a": (latent + 64, hidden),
                            "query_b": (heads * 256, query_rank), "output": (hidden, heads * 256)}.items():
            tensors[attention + role + ".weight"] = projection(shape)
        tensors[attention + "latent.query_embedding.weight"] = query_embedding
        tensors[attention + "latent.output_unembedding.weight"] = output_unembedding
        if layer == 0:
            for role, shape in {"key": (128, hidden), "query": (32 * 128, query_rank),
                                "score": (32, hidden)}.items():
                tensors[attention + "indexer." + role + ".weight"] = projection(shape)
            tensors[attention + "indexer.key_norm.weight"] = np.zeros(128, np.float32)
            tensors[attention + "indexer.key_norm.bias"] = np.zeros(128, np.float32)
        if layer == 0:
            tensors[root + "mlp.experts.gate_up.weight"] = headwise((3, 32, hidden))
            tensors[root + "mlp.experts.down.weight"] = headwise((3, hidden, 16))
            tensors[root + "mlp.router.weight"] = weight((3, hidden))
            tensors[root + "mlp.router.bias"] = weight((3,))
            for role, shape in {"gate": (16, hidden), "up": (16, hidden), "down": (hidden, 16)}.items():
                tensors[root + "mlp.shared_expert." + role + ".weight"] = weight(shape)
        else:
            for role in ("gate", "up", "down"):
                tensors[root + "mlp." + role + ".weight"] = projection((hidden, hidden))
    graph = dict(schema_version=1, architecture="glm_dsa",
                 canonical_naming=dict(namespace="mfq.tensor", version=1, component_roots=["model"]),
                 topology=dict(text_layers=2), graph=dict(kind="causal_lm", backbone="glm_dsa"),
                 components=[dict(kind="text", tensor_root="model", implementation="glm_dsa", policy="decoder")],
                 capabilities=["text"])
    tensors[MODEL_CONFIG_ASSET] = json.dumps(dict(model_type="glm_dsa", text_config=config)).encode()
    tensors[MODEL_GRAPH_ASSET] = json.dumps(graph).encode()
    io.save(path, FileHeader(version=2, model_arch="glm_dsa"), tensors)


async def _probe(tmp_path, model, limit, concurrent, expect_batching):
    executable = os.environ.get("MFQ_ADMISSION_CUDA_RUNTIME")
    if not executable:
        pytest.skip("MFQ_ADMISSION_CUDA_RUNTIME must point to CUDA mfq-runtime")
    tokenizer = _worker_tokenizer(tmp_path)
    environment = dict(os.environ, MFQ_CUDA_VRAM_LIMIT_GIB=str(limit),
                       MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.01", MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1")
    with (tmp_path / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            executable, "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio",
            "--ctx-size", "8192", "--continuous-batching", "8",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
        client = None
        observed = []
        try:
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            before = await client.status()
            assert before["batch_capacity"] == (8 if expect_batching else 1)

            async def generate(index):
                prompt_tokens = 40 + 200 * index if concurrent else 9
                async with client.generate(dict(
                    messages=[dict(role="user", content="ab")], mfq_preformatted_prompt="ab",
                    mfq_benchmark_prompt_tokens=prompt_tokens,
                    temperature=0, max_tokens=2, enable_mtp=False, stream=True,
                    stream_options=dict(include_usage=True), mfq_prefix_cache_enabled=False,
                )) as events:
                    chunks = [event async for event in events if event]
                assert chunks and not any("error" in event for event in chunks), chunks
                usage = [event["usage"] for event in chunks if event.get("usage")]
                assert usage and usage[-1]["completion_tokens"] == 2, chunks
                assert usage[-1]["prompt_tokens"] == prompt_tokens, usage
                return usage[-1]

            async def monitor():
                while True:
                    observed.append(await client.status())
                    await asyncio.sleep(.01)

            watcher = asyncio.create_task(monitor())
            try:
                results = await asyncio.gather(*(generate(i) for i in range(8 if concurrent else 1)))
            finally:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
            after = await _fresh_idle_status(client)
            # Transport metrics include queued requests; the native executor
            # regression checks how many requests actually own model state.
            peak = max((row.get("active_requests", 0) for row in observed), default=0)
            assert after["active_requests"] == 0 and process.returncode is None
            assert after.get("continuous_batching_admission_failures", 0) == 0
            (tmp_path / "result.json").write_text(json.dumps(dict(
                before=before, after=after, peak_transport_requests=peak,
                completed=len(results), usage=results), indent=2))
        finally:
            if client is not None:
                with suppress(Exception):
                    await client.aclose()
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 10)


@pytest.mark.parametrize("family", ["qwen35", "qwen4_exp"])
@pytest.mark.parametrize("limit", [1, 1.5])
def test_single_short_request_under_small_vram_limit(tmp_path, family, limit):
    if not os.environ.get("MFQ_ADMISSION_CUDA_RUNTIME"):
        pytest.skip("CUDA runtime required")
    if family == "qwen35":
        model = tmp_path / "qwen35.mfq"
        _recurrent_model(model)
        header, weights = io.load(model)
        config = json.loads(weights[MODEL_CONFIG_ASSET])
        config["max_position_embeddings"] = 8192
        weights[MODEL_CONFIG_ASSET] = json.dumps(config).encode()
        weights["model.output.weight"].fill(0)
        io.save(model, header, weights)
    else:
        model = _models(tmp_path, qsa=True, context_size=8192)[1]
        header, weights = io.load(model)
        weights["model.output.weight"].fill(0)
        io.save(model, header, weights)
    asyncio.run(asyncio.wait_for(_probe(tmp_path, model, limit, False, True), 120))


def test_serial_capacity_queues_eight_glm_dsa_requests(tmp_path):
    if not os.environ.get("MFQ_ADMISSION_CUDA_RUNTIME"):
        pytest.skip("CUDA runtime required")
    model = tmp_path / "glm-dsa.mfq"
    _glm_dsa_model(model)
    asyncio.run(asyncio.wait_for(_probe(tmp_path, model, 1, True, False), 180))
