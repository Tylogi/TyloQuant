"""Exercise the native Qwen4 image adapter with NINT vision weights and MTP."""
import asyncio
import base64
import json
import os
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest

from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import BackendError, StdioRuntimeClient
from mfq.server.vision.decode import _tensor, _binary_tensors
from tests.test_cuda_flash_next_runtime import mtp_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("bits", [0, 4])
def test_cuda_qwen4_image_and_mtp(tmp_path, bits, _fault_after=0, _fault_scope=1):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("set MFQ_CUDA_RESOURCE_TEST_RUNTIME to a CUDA mfq-runtime")
    config, weights = mtp_fixture("qwen", 1)
    config.update(image_token_id=5, video_token_id=6, vision_config=dict(
        hidden_size=8, intermediate_size=16, depth=1, num_heads=2, patch_size=2,
        temporal_patch_size=1, spatial_merge_size=2, out_hidden_size=16,
        num_position_embeddings=16, hidden_act="gelu_pytorch_tanh"))
    config["text_config"]["max_position_embeddings"] = 128
    rng = np.random.default_rng(91)
    for name, shape in {
        "patch_embedding": (8, 3, 1, 2, 2), "position_embedding": (16, 8),
        "block.0.attention.qkv": (24, 8), "block.0.attention.output": (8, 8),
        "block.0.mlp.up": (16, 8), "block.0.mlp.down": (8, 16),
        "merger.mlp.up": (32, 32), "merger.mlp.down": (16, 32),
    }.items():
        value = rng.normal(0, .05, shape).astype(np.float32)
        weights[f"vision.{name}.weight"] = quantize(value, NintSpec(8, 8, 6)) if len(shape) == 2 and name != "position_embedding" else value
        if name != "position_embedding":
            weights[f"vision.{name}.bias"] = np.zeros(shape[0], np.float32)
    for name in ("block.0.norm1", "block.0.norm2", "merger.norm"):
        weights[f"vision.{name}.weight"] = np.ones(8, np.float32)
        weights[f"vision.{name}.bias"] = np.zeros(8, np.float32)
    graph = dict(schema_version=1, architecture="qwen4_exp",
        canonical_naming=dict(namespace="mfq.tensor", version=1, component_roots=["model", "predictor", "vision"]),
        topology=dict(text_layers=2, predictor_layers=1, vision_layers=1),
        graph=dict(kind="causal_lm", backbone="qwen4_exp"), capabilities=["text", "vision", "mtp"],
        components=[dict(kind="text", tensor_root="model", implementation="qwen4_exp", policy="decoder"),
            dict(kind="predictor", tensor_root="predictor", implementation="next_token_prediction", policy="optional"),
            dict(kind="vision", tensor_root="vision", implementation="grid_vit", policy="optional",
                input_contract="grid_vision.v1", position_policy="grid_mrope")])
    model = tmp_path / "vision.mfq"
    save(model, FileHeader(version=2, model_arch="qwen4_exp"), {**weights,
        MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})
    tokenizer = _worker_tokenizer(tmp_path, cacheable_tokens=True)
    media = dict(version=3, processor="grid_vision.v1",
        pixel_values=_tensor(rng.normal(0, 1, (16, 12)).astype(np.float32), "float32"),
        vision_grid_thw=_tensor(np.array([[1, 4, 4]], np.int32), "int32"),
        image_grid_thw=_tensor(np.array([[1, 4, 4]], np.int32), "int32"),
        vision_types=_tensor(np.array([1], np.int32), "int32"))

    async def run(resident):
        with (tmp_path / f"worker-{resident}.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(str(Path(executable).resolve()),
                "--model", str(model), "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "128",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                env=dict(os.environ, MFQ_KV_TURBOQUANT_BITS=str(bits), MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
                    MFQ_CUDA_VISION_RESIDENT=str(int(resident)),
                    MFQ_TEST_CUDA_PREFILL_FAIL_AFTER=str(_fault_after if not resident else 0),
                    MFQ_TEST_CUDA_PREFILL_FAIL_COUNT="1",
                    MFQ_TEST_CUDA_PREFILL_FAIL_SCOPE=str(_fault_scope),
                    MFQ_QWEN4_WARMUP_MTP="0" if _fault_after else os.environ.get("MFQ_QWEN4_WARMUP_MTP", "1"),
                    MFQ_MTP_HOST_PROMPT_HISTORY=str(int(not resident))))
            client = StdioRuntimeClient(process, control_timeout_seconds=20)
            try:
                status = await client.status()
                assert status["vision_available"] == status["mtp_available"] == 1
                async def generate(mtp, image=media):
                    payload = dict(messages=[dict(role="user", content="image")],
                        mfq_preformatted_prompt="token1" + "token5" * 4 + "token2" * 12,
                        mfq_multimodal=image, enable_vision=True, enable_mtp=mtp,
                        max_tokens=6, temperature=0, stream=True)
                    async with client.generate(payload) as events:
                        output = [event async for event in events if event]
                    metrics = next(event["mfq_metrics"] for event in output if "mfq_metrics" in event)
                    assert metrics["mtp_used"] == mtp
                    return "".join(c.get("delta", {}).get("content", "") for event in output for c in event.get("choices", []))
                outputs = [await generate(False), await generate(True)]
                with pytest.raises(BackendError):
                    await generate(False, {**media, "vision_types": _tensor(np.array([2], np.int32), "int32")})
                outputs.append(await generate(False))
                binary, path = _binary_tensors([
                    (name, np.frombuffer(base64.b64decode(value['data_base64']),
                        dtype=value['dtype']).reshape(value['shape']), value['dtype'])
                    for name, value in media.items() if isinstance(value, dict)])
                unicode_path = path.with_name('mfq-multimodal-视觉-' + path.name.removeprefix('mfq-multimodal-'))
                path.rename(unicode_path)
                binary['binary_file']['path'] = str(unicode_path)
                binary.update(version=media['version'], processor=media['processor'])
                try:
                    assert await generate(False, binary) == outputs[0]
                    assert await generate(True, binary) == outputs[1]
                    for invalid in ({'token': '00'*32}, {'size': binary['binary_file']['size']+64}):
                        with pytest.raises(BackendError):
                            await generate(False, {**binary, 'binary_file': {**binary['binary_file'], **invalid}})
                finally:
                    unicode_path.unlink()
                after = await client.status()
                assert after['resident_weight_bytes'] == status['resident_weight_bytes']
                return outputs, status['resident_weight_bytes']
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    expected, resident_bytes = asyncio.run(run(True))
    actual, streamed_bytes = asyncio.run(run(False))
    assert actual == expected
    assert streamed_bytes < resident_bytes


def test_cuda_vision_prefill_reclaims_and_retries(tmp_path):
    test_cuda_qwen4_image_and_mtp(tmp_path, 0, _fault_after=5)
    trace = (tmp_path / "worker-False.log").read_text(errors="replace")
    assert trace.count("runtime_prefill_retry scope=vision") == 1, trace
    assert "cleanup_error" not in trace


def test_cuda_prepared_image_prefill_replays_positions_and_embeddings(tmp_path):
    test_cuda_qwen4_image_and_mtp(tmp_path, 0, _fault_after=1, _fault_scope=6)
    trace = (tmp_path / "worker-False.log").read_text(errors="replace")
    assert "runtime_prefill_retry scope=request" in trace, trace
    assert "cleanup_error" not in trace
