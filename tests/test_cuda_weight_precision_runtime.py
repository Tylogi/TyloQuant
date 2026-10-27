"""Native CUDA loading and generation across legal floating weight layouts."""

import asyncio
import copy
import json
import os
import re
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import BFloat16Array, Float8E4M3Array, open_mmap, save
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import StdioRuntimeClient
from tests.cuda_weight_precision_fixtures import gemma4_fixture, glm_dsa_fixture
from tests.test_cuda_flash_next_runtime import glm_model_fixture, qwen_model_fixture
from tests.test_cuda_kv_quantization_native import write_qwen35_fixture
from tests.test_cuda_nemotron_runtime import model_fixture, write_nemotron_fixture
from tests.test_native_nint_ple import _worker_tokenizer


ARCHITECTURES = ("gemma4", "glm_dsa", "qwen4_exp", "glm5_next", "nemotron_h", "qwen3_5_moe")


def native(value, dtype):
    value = np.asarray(value, np.float32)
    if dtype == "BF16":
        bits = value.view(np.uint32)
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        return (rounded >> 16).astype(np.uint16).view(BFloat16Array)
    return value.astype(np.float16 if dtype == "F16" else np.float32)


def fixture(architecture, path):
    if architecture == "gemma4":
        config, weights = gemma4_fixture()
    elif architecture == "glm_dsa":
        config, weights = glm_dsa_fixture()
    elif architecture == "qwen4_exp":
        config, weights = qwen_model_fixture(2, False, False)
    elif architecture == "glm5_next":
        config, weights = glm_model_fixture(("linear_attention", "deepseek_sparse_attention"), True)
    elif architecture == "nemotron_h":
        config, weights = model_fixture(("moe", "attention", "mlp"))
    else:
        write_qwen35_fixture(path, context_size=64)
        with open_mmap(path) as store:
            config = json.loads(store.get(MODEL_CONFIG_ASSET))
            weights = {name: store.get(name) for name in store if not name.startswith("__")}
        weights = {name: value for name, value in weights.items() if isinstance(value, np.ndarray)}
        if architecture == "qwen3_5_moe":
            config.update(num_experts=3, num_experts_per_tok=2, moe_intermediate_size=16,
                          shared_expert_intermediate_size=16)
            rng = np.random.default_rng(3435)
            for layer in range(2):
                prefix = f"model.block.{layer}.mlp."
                for role in ("gate", "up", "down"):
                    weights.pop(prefix + role + ".weight")
                for role, shape in {"experts.gate_up": (3, 32, 32), "experts.down": (3, 32, 16),
                                    "router": (3, 32), "shared_expert.gate": (16, 32),
                                    "shared_expert.up": (16, 32), "shared_expert.down": (32, 16),
                                    "shared_expert.router": (1, 32)}.items():
                    weights[prefix + role + ".weight"] = rng.normal(0, 0.05, shape).astype(np.float32)
    text = config.get("text_config", config.get("llm_config", config))
    text["max_position_embeddings"] = 64
    text["eos_token_id"] = []
    config["eos_token_id"] = []
    # The reference and variants share weights; only their storage changes.
    weights = {name: native(value, "F16")
               if isinstance(value, np.ndarray) and value.ndim >= 2 else value
               for name, value in weights.items()}
    return config, weights


def write_fixture(path, architecture, config, weights):
    if architecture == "nemotron_h":
        write_nemotron_fixture(path, config, weights)
        return
    architecture = architecture.removesuffix("_moe")
    text = config.get("text_config", config.get("llm_config", config))
    graph = dict(schema_version=1, architecture=architecture,
                 canonical_naming=dict(namespace="mfq.tensor", version=1, component_roots=["model"]),
                 topology=dict(text_layers=text["num_hidden_layers"]),
                 graph=dict(kind="causal_lm", backbone=architecture), capabilities=["text"],
                 components=[dict(kind="text", tensor_root="model", implementation=architecture,
                                  policy="decoder")])
    save(path, FileHeader(version=2, model_arch=architecture), {
        **weights, MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})


def variant_weights(weights, kind):
    result = copy.copy(weights)
    for name, value in weights.items():
        if not isinstance(value, np.ndarray) or value.ndim < 2:
            continue
        expert = ".mlp.experts." in name
        if kind.startswith("mfe-layer-") and expert:
            resident = int(name.split(".")[2]) == int(kind.rsplit("-", 1)[1])
            matrices = [(name, value)]
            if not resident and name.endswith(".gate_up.weight"):
                result.pop(name)
                matrices = [(name.replace("gate_up", role), matrix)
                            for role, matrix in zip(("gate", "up"), np.split(value, 2, axis=1), strict=True)]
            for target, matrix in matrices:
                count, output, width = matrix.shape
                flat = matrix.reshape(count * output, width)
                tensor = native(flat, "BF16") if resident else quantize(flat, NintSpec(8, 8, 6))
                result[target] = MfeTensor(matrix.shape, (MfePool(np.arange(count, dtype=np.int32), tensor),))
            continue
        if kind == "mfe-split" and expert:
            if name.endswith(".gate_up.weight"):
                gate, up = np.split(value, 2, axis=1)
                result.pop(name)
                result[name.replace("gate_up", "gate")] = native(gate, "BF16")
                name, value = name.replace("gate_up", "up"), up
            count, output, width = value.shape
            pool = MfePool(np.arange(count, dtype=np.int32), native(value.reshape(count * output, width), "F16"))
            result[name] = MfeTensor(value.shape, (pool,))
            continue
        if kind.startswith("dense-") and not expert:
            result[name] = native(value, kind.removeprefix("dense-"))
        elif kind.startswith("bank-") and expert:
            result[name] = native(value, kind.removeprefix("bank-"))
        elif kind.startswith("mfe-") and expert:
            if kind in ("mfe-bank", "mfe-nint-bank") and name.endswith(".down.weight"):
                result[name] = native(value, "BF16")
                continue
            if kind in ("mfe-bank-reverse", "mfe-bank-nint") and not name.endswith(".down.weight"):
                result[name] = native(value, "BF16")
                continue
            count, output, width = value.shape
            pools = []
            for index in range(count):
                if kind in ("mfe-nint-bank", "mfe-bank-nint"):
                    tensor = quantize(value[index], NintSpec(8, 8, 6))
                elif kind == "mfe-mixed":
                    tensor = (quantize(value[index], NintSpec(8, 8, 6)) if index == 0
                              else native(value[index], "F16" if index == 1 else "BF16"))
                else:
                    dtype = "F16" if kind.startswith("mfe-bank") else kind.removeprefix("mfe-")
                    tensor = native(value[index], dtype)
                pools.append(MfePool(np.array([index], np.int32), tensor))
            # Nontrivial ownership/order catches accidental global/local expert indexing.
            result[name] = MfeTensor((count, output, width), tuple(reversed(pools)))
    return result


async def generate(executable, model, tokenizer, log_path, *, cache=False):
    environment = dict(os.environ, MFQ_CUDA_VRAM_LIMIT_GIB="2", MFQ_EMBEDDING_MAPPED="0",
                       MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1", MFQ_MOE_PIPELINE="1",
                       MFQ_CUDA_WORKSPACE_RESERVE_GIB="0.01")
    arguments = [str(executable), "--model", str(model), "--tokenizer", str(tokenizer),
                 "--transport", "stdio", "--ctx-size", "64"]
    if cache:
        arguments += ["--moe-gpu-cache-gb", "0.01"]
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
                                                       stdout=asyncio.subprocess.PIPE, stderr=log, env=environment)
        client = StdioRuntimeClient(process, control_timeout_seconds=30)
        try:
            await client.status()
            outputs = []
            for prompt_tokens in (1, 9):
                payload = dict(messages=[dict(role="user", content="ab")], mfq_preformatted_prompt="ab",
                               mfq_benchmark_prompt_tokens=prompt_tokens, max_tokens=4, temperature=0,
                               enable_mtp=False, mfq_prefix_cache_enabled=False,
                               stream=True, stream_options=dict(include_usage=True))
                async with client.generate(payload) as events:
                    chunks = [event async for event in events if event]
                assert chunks and not any("error" in event for event in chunks), chunks
                usage = next(event["usage"] for event in chunks if event.get("usage"))
                assert 1 <= usage["completion_tokens"] <= 4, chunks
                text = "".join(choice.get("delta", {}).get("content", "")
                               for event in chunks for choice in event.get("choices", []))
                assert "\ufffd" not in text, chunks
                if not text:
                    assert usage["completion_tokens"] == 1, chunks
                outputs.append(re.findall(r"token\d+|<[^>]+>|.", text))
            return outputs
        except Exception as error:
            raise AssertionError(f"{error}\n{log_path.read_text(errors='replace')}") from error
        finally:
            with suppress(Exception):
                await client.aclose()
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 10)


@pytest.fixture
def runtime():
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a native CUDA mfq-runtime")
    return Path(executable).resolve()


@pytest.mark.parametrize("architecture", ARCHITECTURES + ("qwen3_5",))
@pytest.mark.parametrize("kind", ("dense-F32", "dense-BF16"))
def test_dense_storage_preserves_activation_contract(tmp_path, runtime, architecture, kind):
    check_generation(tmp_path, runtime, architecture, kind)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("kind", ("bank-F32", "bank-BF16", "mfe-F16", "mfe-BF16", "mfe-mixed",
                                 "mfe-bank", "mfe-bank-reverse", "mfe-nint-bank", "mfe-bank-nint"))
def test_expert_storage_generation(tmp_path, runtime, architecture, kind):
    check_generation(tmp_path, runtime, architecture, kind)


@pytest.mark.parametrize("architecture", ("qwen4_exp", "glm5_next", "qwen3_5_moe"))
def test_split_expert_projections_mix_native_and_mfe(tmp_path, runtime, architecture):
    check_generation(tmp_path, runtime, architecture, "mfe-split")


@pytest.mark.parametrize("resident_layer", (0, 1))
def test_qwen4_exp_mix_resident_and_cached_layers(tmp_path, runtime, resident_layer):
    check_generation(tmp_path, runtime, "qwen4_exp", f"mfe-layer-{resident_layer}")


def check_generation(tmp_path, runtime, architecture, kind):
    reference, variant = tmp_path / "f16.mfq", tmp_path / "variant.mfq"
    config, weights = fixture(architecture, reference)
    write_fixture(reference, architecture, config, weights)
    write_fixture(variant, architecture, config, variant_weights(weights, kind))
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)

    async def compare():
        baseline = await generate(runtime, reference, tokenizer, tmp_path / "f16.log")
        actual = await generate(runtime, variant, tokenizer, tmp_path / "variant.log", cache=kind.startswith("mfe-"))
        (tmp_path / "greedy.json").write_text(json.dumps({
            "architecture": architecture, "storage": kind, "prompt_tokens": [1, 9],
            "f16": baseline, "actual": actual,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        # F32/F16 use the same half-representable weights. BF16 and the mixed
        # 8-bit cohort may change one of four printable vocabulary pieces.
        for expected, observed in zip(baseline, actual, strict=True):
            if kind in ("dense-F32", "bank-F32", "mfe-F16"):
                assert observed == expected
            else:
                assert len(expected) == len(observed), (expected, observed)
                matches = sum(left == right for left, right in zip(expected, observed, strict=True))
                assert matches >= (3 * len(expected) + 3) // 4, (expected, observed)

    asyncio.run(compare())


@pytest.mark.parametrize("routed_input", (False, True))
@pytest.mark.parametrize("cached", (False, True))
def test_native_mfe_cohort_values(tmp_path, runtime, routed_input, cached):
    import subprocess
    rng = np.random.default_rng(3401)
    source = rng.normal(0, 0.05, (4, 4, 32)).astype(np.float32)
    half_ids, bf16_ids = np.array([2, 0], np.int32), np.array([3, 1], np.int32)
    half = native(source[half_ids].reshape(8, 32), "F16")
    brain = native(source[bf16_ids].reshape(8, 32), "BF16")
    tensor = MfeTensor(source.shape, (MfePool(half_ids, half), MfePool(bf16_ids, brain)))
    model = tmp_path / "cohorts.mfq"
    save(model, FileHeader(version=2, model_arch="moe-cache-test"), {"experts.weight": tensor})
    command = [str(runtime.with_name("mfq-diagnostics" + runtime.suffix)), "--model", str(model),
               "--check-mfe-tensor", "experts.weight", "--check-mfe-tokens", "9",
               "--check-mfe-routes", "2", "--check-mfe-reps", "1"]
    if routed_input:
        command.append("--check-mfe-routed-input")
    if cached:
        command += ["--moe-gpu-cache-gb", "0.01"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45,
                            env=dict(os.environ, MFQ_CUDA_VRAM_LIMIT_GIB="2"))
    assert result.returncode == 0, result.stdout + result.stderr
    row = next(line for line in result.stdout.splitlines() if line.startswith("mfe_tensor_check "))
    values = np.fromstring(row.split(" values=", 1)[1], sep=",").reshape(9, 2, 4)
    assert np.isfinite(values).all()
    bank = np.empty_like(source)
    bank[half_ids] = half.astype(np.float32).reshape(2, 4, 32)
    bank[bf16_ids] = (brain.astype(np.uint32) << 16).view(np.float32).reshape(2, 4, 32)
    sequence = np.arange(9 * (2 if routed_input else 1) * 32, dtype=np.float32)
    inputs = ((sequence % 257 - 128) / 127 + np.float32(0.03125) *
              np.sin(sequence * np.float32(0.015625))).astype(np.float16).astype(np.float32)
    inputs = inputs.reshape(9, 2, 32) if routed_input else inputs.reshape(9, 32)
    expected = np.empty((9, 2, 4), np.float32)
    for token in range(9):
        for route in range(2):
            expert = (token * 2 + route * 3) % 4
            activation = inputs[token, route] if routed_input else inputs[token]
            expected[token, route] = bank[expert] @ activation
    # BF16 operands and projection output each round at their native boundary.
    np.testing.assert_allclose(values, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("kind", ("dense-fp8", "mfe-mxfp8", "embedding-sq"))
def test_unsupported_storage_rejected_during_load(tmp_path, runtime, kind):
    from mfq.formats.mx import MxTensor
    config, weights = fixture("gemma4", tmp_path / "unused.mfq")
    if kind == "dense-fp8":
        name = "model.block.0.attention.output.weight"
        weights[name] = np.zeros(weights[name].shape, np.uint8).view(Float8E4M3Array)
        dtype = "F8_E4M3"
    elif kind == "embedding-sq":
        from mfq.quantize.mxfp4_sq import quantize_mxfp4_sq
        name = "model.token_embedding.weight"
        rows, width = weights[name].shape
        weights[name] = quantize_mxfp4_sq(np.zeros((rows, width // 2), np.uint8),
                                        np.full((rows, width // 32), 127, np.uint8), row_q_bits=4)
        dtype = "MXFP4-SQ"
    else:
        name = "model.block.0.mlp.experts.gate_up.weight"
        count, output, width = weights[name].shape
        tensor = MxTensor("MXFP8", (count * output, width),
                          np.zeros((count * output, width), np.uint8),
                          np.full((count * output, width // 32), 127, np.uint8))
        weights[name] = MfeTensor((count, output, width), (MfePool(np.arange(count, dtype=np.int32), tensor),))
        dtype = "MXFP8"
    model = tmp_path / "unsupported.mfq"
    write_fixture(model, "gemma4", config, weights)
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)
    import subprocess
    result = subprocess.run([str(runtime), "--model", str(model), "--tokenizer", str(tokenizer),
                             "--transport", "stdio", "--ctx-size", "64"],
                            input="", capture_output=True, text=True, timeout=45,
                            env=dict(os.environ, MFQ_CUDA_VRAM_LIMIT_GIB="2"))
    assert result.returncode != 0
    assert name in result.stderr and dtype in result.stderr, result.stderr
    assert "quantize" in result.stderr or "--dense-dtype" in result.stderr, result.stderr
