import os
from datetime import datetime, timezone
from dataclasses import replace

import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.protocol.models import ModelArtifactResource, ModelLoadRequest
from mfq.server.runtime.native import supports_cuda_ram_expert_pipeline
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.services.hardware import GpuMemory, HardwareIdentity
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog
from mfq.server.services.jobs import JobExecutionError
from mfq.server.services.model_memory import cache_profile, checkpoint_cache_profile
from mfq.formats.nvq import E8_256, NVQ2_E8, NvqJscTensor


def make_nvq_fixture():
    return NvqJscTensor((6, 96), 0, 96, np.ones(6, dtype=np.float32),
        np.arange(16, dtype=np.float32), np.zeros(16, dtype=np.uint8),
        np.ones((6, 4), dtype=np.uint8), np.zeros((6, 12), dtype=np.uint8),
        np.zeros((6, 12), dtype=np.uint8), (E8_256[None].astype(np.int16) * 8).astype(np.int8),
        base_spec=NVQ2_E8)

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _no_host_commit_accounting(monkeypatch):
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.host_commit_capacity", lambda: (None, None)
    )
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.gpu_memory_status", lambda: [])
    monkeypatch.delenv("MFQ_CUDA_VRAM_LIMIT_GIB", raising=False)
    monkeypatch.delenv("MFQ_CUDA_WORKSPACE_RESERVE_GIB", raising=False)


def artifact(tmp_path):
    path = tmp_path / "model.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_nvq_fixture()),))
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": experts})
    resource = ModelArtifactResource(id="1" * 32, name="model", architecture="qwen4_exp",
        total_bytes=48 * GIB, shard_count=1, tensor_count=1, record_count=1,
        complete=True, loadable=True, modified_at=datetime.now(timezone.utc))
    return DiscoveredModel(resource=resource, path=path, routed_expert_bytes=44 * GIB)


@pytest.mark.parametrize("unified,device", [(False, "0"), (False, "1"), (True, "0")])
def test_cuda_auto_expert_budget_uses_selected_physical_device(tmp_path, monkeypatch, unified, device):
    hardware = HardwareIdentity("cpu", 8, gpu_memory=(
        GpuMemory("NVIDIA GB10" if unified else "RTX GPU", 24 * GIB, unified=unified),
        GpuMemory("RTX GPU", 16 * GIB),
    )) if not unified else HardwareIdentity("cpu", 8, gpu_memory=(GpuMemory("NVIDIA GB10", 128 * GIB, unified=True),))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: hardware)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda")
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", device_ids=[device])
    selected = pool._apply_automatic_expert_residency(model, request)
    if unified:
        assert selected.moe_gpu_cache_gb is None
    else:
        capacity = hardware.gpu_memory[int(device)].capacity_bytes
        assert selected.moe_gpu_cache_gb is not None and selected.moe_gpu_cache_gb > 0
        # Native admission prices actual allocations; no extra third for PP.
        assert pool._estimated_load_bytes(model, selected) == capacity - max(GIB, capacity // 16)
    explicit = request.model_copy(update={"moe_gpu_cache_gb": 7})
    assert pool._apply_automatic_expert_residency(model, explicit).moe_gpu_cache_gb == 7


@pytest.mark.parametrize("capacity,configured,limit", [
    (96 * GIB, None, 90 * GIB),
    (24 * GIB, None, 24 * GIB - 24 * GIB // 16),
    (16 * GIB, None, 15 * GIB),
    (96 * GIB, "40", 40 * GIB),
])
def test_cuda_vram_limit_keeps_existing_margin_and_explicit_limit(tmp_path, monkeypatch,
                                                                 capacity, configured, limit):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", capacity),)))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda")
    if configured:
        pool.runtime_environment["MFQ_CUDA_VRAM_LIMIT_GIB"] = configured
    request = ModelLoadRequest(model="model", device_ids=["0"])
    assert pool._cuda_device_vram_limit(request) == limit
    selected = pool._apply_automatic_expert_residency(artifact(tmp_path), request)
    # Automatic residency fills the worker's limit, never the whole device.
    assert pool._estimated_load_bytes(artifact(tmp_path), selected) <= limit


@pytest.mark.parametrize("configured", ["0", "-1", "nan", "inf", "invalid"])
def test_cuda_vram_limit_rejects_invalid_explicit_values(tmp_path, monkeypatch, configured):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 24 * GIB),)))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda")
    pool.runtime_environment["MFQ_CUDA_VRAM_LIMIT_GIB"] = configured
    with pytest.raises(JobExecutionError) as error:
        pool._cuda_device_vram_limit(ModelLoadRequest(model="model", device_ids=["0"]))
    assert error.value.detail.code == "runtime_vram_budget_invalid"


def _headroom_pool(tmp_path, monkeypatch, *, total_gib=89.5, always_streamed_gib=0, pipeline=None):
    # Nominal 96-GB card: reported capacity yields the observed ~89.6-GiB ceiling.
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", int(95.6 * GIB)),)))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (256 * GIB, 200 * GIB))
    profile = cache_profile(dict(model_type="qwen4_exp", max_position_embeddings=1048576,
        num_hidden_layers=64, full_attention_interval=4, num_key_value_heads=8, head_dim=128,
        indexer_head_dim=128, indexer_compress_ratio=8), include_recurrent=False, cuda=True)
    assert profile is not None
    monkeypatch.setattr("mfq.server.services.model_memory.checkpoint_cache_profile", lambda *args, **kwargs: profile)
    if pipeline is not None:
        # The fixture bank is pipeline-capable; reserve tests also cover banks
        # the RAM-expert pipeline cannot serve, which stay resident when they fit.
        monkeypatch.setattr("mfq.server.runtime.runtime_pool.supports_cuda_ram_expert_pipeline",
            lambda path: pipeline)
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    pool.memory_policy.prefix_directory = str(tmp_path / "prefix")
    model = artifact(tmp_path)
    model = replace(model, resource=model.resource.model_copy(update={
        "total_bytes": int(total_gib * GIB), "context_capacity": 1048576,
    }), routed_expert_bytes=int((total_gib - always_streamed_gib - 4) * GIB),
        always_streamed_bytes=int(always_streamed_gib * GIB))
    return pool, model


def test_cuda_borderline_residency_selects_cache_and_fast_launch(tmp_path, monkeypatch, caplog):
    pool, model = _headroom_pool(tmp_path, monkeypatch)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    assert model.resource.total_bytes <= pool._cuda_device_vram_limit(request)
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        selected = pool._apply_automatic_expert_residency(model, request)
    assert selected.moe_gpu_cache_gb > 0
    command, environment = pool._launch_configuration(model, selected, port=1234)
    assert "--moe-gpu-cache-gb" in command
    for name in ("MFQ_MOE_PIPELINE", "MFQ_MOE_TWO_STAGE_FFN", "MFQ_MOE_PREFILL_LAYER",
                 "MFQ_MOE_DECODE_HOST_PIN", "MFQ_MOE_TRANSFER_CACHE"):
        assert environment[name] == "1"
    assert float(environment["MFQ_CUDA_VRAM_LIMIT_GIB"]) * GIB == pool._cuda_device_vram_limit(request)
    assert "decision=cache" in caplog.text.split()
    assert f"total_bytes={model.resource.total_bytes}" in caplog.text
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    for field, value in (("kv_reserve_bytes", reserve.kv_bytes), ("workspace_reserve_bytes", 4 * GIB),
                         ("overhead_reserve_bytes", 1280 << 20), ("runtime_reserve_bytes", reserve.total_bytes),
                         ("ceiling_bytes", pool._cuda_device_vram_limit(request)),
                         ("cache_bytes", int(selected.moe_gpu_cache_gb * GIB))):
        assert f"{field}={value}" in caplog.text


@pytest.mark.parametrize("resident_gib,streamed_gib", [(40, 0), (66, 0), (40, 60)])
def test_cuda_roomy_residency_launches_without_pipeline_or_vram_limit(
    tmp_path, monkeypatch, caplog, resident_gib, streamed_gib,
):
    pool, model = _headroom_pool(tmp_path, monkeypatch, pipeline=False,
        total_gib=resident_gib + streamed_gib, always_streamed_gib=streamed_gib)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        selected = pool._apply_automatic_expert_residency(model, request)
    assert selected is request and selected.moe_gpu_cache_gb is None
    command, environment = pool._launch_configuration(model, selected, port=1234)
    assert "--moe-gpu-cache-gb" not in command
    # Resident workers keep the whole device: the limit belongs to the expert
    # cache fast path (TCC GPUs have no WDDM adapter to enforce it).
    assert "MFQ_CUDA_VRAM_LIMIT_GIB" not in environment
    assert "MFQ_MOE_PIPELINE" not in environment
    assert "MFQ_MOE_HOST_CACHE_BYTES" not in environment
    assert {"decision=resident", "cache_bytes=0"} <= set(caplog.text.split())


@pytest.mark.parametrize("cache_gib", [0, 7])
def test_cuda_explicit_cache_budget_bypasses_automatic_reserve(tmp_path, monkeypatch, cache_gib):
    pool, model = _headroom_pool(tmp_path, monkeypatch)
    def unexpected_reserve(*args):
        pytest.fail("explicit expert residency must bypass automatic reserve estimation")
    monkeypatch.setattr(pool, "_cuda_residency_runtime_reserve", unexpected_reserve)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=cache_gib, device_ids=["0"])
    assert pool._apply_automatic_expert_residency(model, request) is request
    command, environment = pool._launch_configuration(model, request, port=1234)
    assert float(command[command.index("--moe-gpu-cache-gb") + 1]) == cache_gib
    assert (environment.get("MFQ_MOE_PIPELINE") == "1") == (cache_gib > 0)


def test_cuda_residency_reserve_grows_with_resolved_context(tmp_path, monkeypatch):
    pool, model = _headroom_pool(tmp_path, monkeypatch, pipeline=False, total_gib=80)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    short = pool._cuda_residency_runtime_reserve(model, request)
    assert pool._apply_automatic_expert_residency(model, request) is request
    pool.context_policy.model_overrides["model"] = 262144
    long_request = request.model_copy(update={"context_size": None})
    long = pool._cuda_residency_runtime_reserve(model, long_request)
    assert long.context == 262144 and long.kv_bytes > short.kv_bytes
    assert pool._apply_automatic_expert_residency(model, long_request).moe_gpu_cache_gb > 0
    assert pool.resolve_context_size(model, long_request.context_size) == 262144


def test_cuda_residency_includes_runtime_reserve_at_exact_ceiling(tmp_path, monkeypatch):
    pool, model = _headroom_pool(tmp_path, monkeypatch, pipeline=False, total_gib=40)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    required = model.resource.total_bytes + pool._cuda_residency_runtime_reserve(model, request).total_bytes
    assert pool._apply_automatic_expert_residency(model, request, memory_ceiling=required) is request
    assert pool._apply_automatic_expert_residency(model, request,
        memory_ceiling=required - 1).moe_gpu_cache_gb > 0


@pytest.mark.parametrize("architecture,total_gib,kv_source", [
    ("nemotron_h", 66, "config_attention_fp16_min_1gib"), ("qwen3_next", 40, "checkpoint_kv_full_context"),
])
def test_cuda_roomy_checkpoint_metadata_keeps_resident_launch(
    tmp_path, monkeypatch, architecture, total_gib, kv_source,
):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=total_gib)
    monkeypatch.setattr("mfq.server.services.model_memory.checkpoint_cache_profile", checkpoint_cache_profile)
    config = dict(model_type=architecture, num_hidden_layers=64, max_position_embeddings=262144,
        full_attention_interval=4, num_key_value_heads=8, head_dim=128)
    save(model.path, FileHeader(version=2, model_arch=architecture, extra={"hf_config": config}),
        {"weight": np.zeros((2, 2), dtype=np.float16)})
    model = replace(model, resource=model.resource.model_copy(update={"architecture": architecture}))
    request = ModelLoadRequest(model="model", context_size=32768, device_ids=["0"])
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    # 16 attention layers x 8 KV heads x 128 x FP16 K/V for 32K tokens.
    assert reserve.kv_source == kv_source and reserve.kv_bytes == 2 * GIB
    assert model.resource.total_bytes + reserve.total_bytes < pool._cuda_device_vram_limit(request)
    assert pool._apply_automatic_expert_residency(model, request) is request
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert "MFQ_CUDA_VRAM_LIMIT_GIB" not in environment
    assert "MFQ_MOE_PIPELINE" not in environment


@pytest.mark.parametrize("placement,quantized", [("auto", False), ("auto_ssd", False),
    ("ram", False), ("vram", False), ("auto", True), ("vram", True)])
def test_cuda_residency_reserve_honors_kv_policy(tmp_path, monkeypatch, placement, quantized):
    from mfq.server.runtime.kv_budget import checkpoint_kv_budget

    pool, model = _headroom_pool(tmp_path, monkeypatch, pipeline=False, total_gib=80)
    request = ModelLoadRequest(model="model", context_size=262144, device_ids=["0"],
        kv_memory={"placement": placement}, kv_quantization={"enabled": quantized, "bits": 4})
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    plan = checkpoint_kv_budget(model, 262144, request.kv_memory, request.kv_quantization)
    assert reserve.kv_bytes == plan.gpu_budget_bytes
    assert reserve.kv_source == "checkpoint_kv_budget"
    selected = pool._apply_automatic_expert_residency(model, request)
    assert (selected.moe_gpu_cache_gb is None) == (placement == "ram" or (placement == "auto" and quantized))


@pytest.mark.parametrize("allocation", ["preallocated", "dynamic"])
def test_cuda_residency_reserve_honors_explicit_kv_budget(tmp_path, monkeypatch, allocation):
    pool, model = _headroom_pool(tmp_path, monkeypatch)
    request = ModelLoadRequest(model="model", context_size=4096,
        kv_memory={"allocation": allocation, "gpu_budget_bytes": 8 * GIB})
    assert pool._cuda_residency_runtime_reserve(model, request).kv_bytes == 8 * GIB


@pytest.mark.parametrize("configured", ["0", "1.5", "8"])
@pytest.mark.parametrize("inherited", [False, True])
def test_cuda_residency_reserve_honors_workspace_environment(tmp_path, monkeypatch, configured, inherited):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=80)
    if inherited:
        monkeypatch.setenv("MFQ_CUDA_WORKSPACE_RESERVE_GIB", configured)
    else:
        monkeypatch.setenv("MFQ_CUDA_WORKSPACE_RESERVE_GIB", "16")
        pool.runtime_environment["MFQ_CUDA_WORKSPACE_RESERVE_GIB"] = configured
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    assert pool._cuda_residency_runtime_reserve(model, request).workspace_bytes == int(float(configured) * GIB)


def test_cuda_workspace_override_can_require_expert_cache(tmp_path, monkeypatch):
    pool, model = _headroom_pool(tmp_path, monkeypatch, pipeline=False, total_gib=80)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    assert pool._apply_automatic_expert_residency(model, request) is request
    pool.runtime_environment["MFQ_CUDA_WORKSPACE_RESERVE_GIB"] = "16"
    assert pool._apply_automatic_expert_residency(model, request).moe_gpu_cache_gb > 0


@pytest.mark.parametrize("configured", ["-1", "nan", "inf", "invalid", ""])
def test_cuda_residency_rejects_invalid_workspace_environment(tmp_path, monkeypatch, configured):
    pool, model = _headroom_pool(tmp_path, monkeypatch)
    pool.runtime_environment["MFQ_CUDA_WORKSPACE_RESERVE_GIB"] = configured
    with pytest.raises(JobExecutionError) as error:
        pool._apply_automatic_expert_residency(model, ModelLoadRequest(model="model", device_ids=["0"]))
    assert error.value.detail.code == "runtime_workspace_budget_invalid"


def test_cuda_residency_missing_profile_uses_context_dependent_fallback(tmp_path, monkeypatch, caplog):
    pool, model = _headroom_pool(tmp_path, monkeypatch, pipeline=False, total_gib=66)
    monkeypatch.setattr("mfq.server.services.model_memory.checkpoint_cache_profile", lambda *args, **kwargs: None)
    request = ModelLoadRequest(model="model", context_size=32768, device_ids=["0"])
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    assert reserve.kv_bytes == 8 * GIB
    assert pool._apply_automatic_expert_residency(model, request) is request
    long_request = request.model_copy(update={"context_size": 262144})
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        assert pool._apply_automatic_expert_residency(model, long_request).moe_gpu_cache_gb > 0
    assert "kv_source=fallback_256kib_per_token_min_1gib_max_ceiling" in caplog.text.split()
    assert pool._cuda_residency_runtime_reserve(model, long_request).kv_bytes == 64 * GIB


@pytest.mark.parametrize("backend,format", [("metal", "mfq"), ("metal", "hf"), ("cuda", "hf")])
def test_runtime_headroom_does_not_change_hf_or_metal_residency(tmp_path, monkeypatch, backend, format):
    pool, model = _headroom_pool(tmp_path, monkeypatch)
    pool.backend = backend
    model = replace(model, resource=model.resource.model_copy(update={"format": format}))
    pool.max_runtime_memory_bytes = 90 * GIB
    def unexpected_reserve(*args):
        pytest.fail("CUDA MFQ runtime reserve must not affect HF or Metal residency")
    monkeypatch.setattr(pool, "_cuda_residency_runtime_reserve", unexpected_reserve)
    request = ModelLoadRequest(model="model", device_ids=["0"])
    selected = pool._apply_automatic_expert_residency(model, request)
    assert selected.moe_gpu_cache_gb == (0 if format == "hf" else None)


@pytest.mark.parametrize("inherited", [False, True])
def test_cuda_resident_launch_preserves_explicit_vram_limit(tmp_path, monkeypatch, inherited):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    if inherited:
        monkeypatch.setenv("MFQ_CUDA_VRAM_LIMIT_GIB", "76")
    else:
        monkeypatch.setenv("MFQ_CUDA_VRAM_LIMIT_GIB", "70")
        pool.runtime_environment["MFQ_CUDA_VRAM_LIMIT_GIB"] = "76"
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert environment["MFQ_CUDA_VRAM_LIMIT_GIB"] == "76"


@pytest.mark.parametrize("configured", ["0", "nan"])
def test_cuda_explicit_cache_launch_keeps_vram_limit_validation(tmp_path, monkeypatch, configured):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    pool.runtime_environment["MFQ_CUDA_VRAM_LIMIT_GIB"] = configured
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=7, device_ids=["0"])
    with pytest.raises(JobExecutionError) as error:
        pool._launch_configuration(model, request, port=1234)
    assert error.value.detail.code == "runtime_vram_budget_invalid"


def test_cuda_unified_launch_does_not_add_discrete_vram_limit(tmp_path, monkeypatch):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("NVIDIA GB10", 128 * GIB, unified=True),)))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "unified"})
    request = ModelLoadRequest(model="model", device_ids=["0"])
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert "MFQ_CUDA_VRAM_LIMIT_GIB" not in environment


@pytest.mark.parametrize("available,enabled", [(52 * GIB, True), (8 * GIB, False)])
def test_cuda_launch_selects_ram_or_ssd_expert_cache(tmp_path, monkeypatch, available, enabled):
    monkeypatch.delenv("MFQ_MODEL_FILE_IO", raising=False)
    monkeypatch.delenv("MFQ_PLE_FILE_MAP", raising=False)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (64 * GIB, available))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 24 * GIB),)))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    pool.memory_policy.prefix_directory = str(tmp_path / "prefix")
    model = artifact(tmp_path)
    assert supports_cuda_ram_expert_pipeline(model.path)
    # Select the mocked GPU explicitly, independent of the test runner's
    # CUDA_VISIBLE_DEVICES setting.
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    command, environment = pool._launch_configuration(model, request, port=1234)
    assert "--moe-gpu-cache-gb" in command
    assert (environment.get("MFQ_MOE_PRELOAD_ALL") == "1") == enabled
    assert environment["MFQ_MOE_SSD_EXPERTS"] == "1"
    assert environment["MFQ_MOE_ASSERT_RESIDENT"] == str(int(enabled))
    if not enabled:
        assert int(environment["MFQ_MOE_HOST_CACHE_BYTES"]) == 6 * GIB
        assert environment["MFQ_MOE_NVQ_DENSE"] == "0"
        assert environment["MFQ_MODEL_FILE_IO"] == "direct"
        assert environment["MFQ_PLE_FILE_MAP"] == "1"
    if enabled:
        assert environment["MFQ_MOE_PIPELINE"] == "1"
        assert environment["MFQ_MOE_TWO_STAGE_FFN"] == "1"
        assert environment["MFQ_MOE_RAM_PCIE"] == "1"
        assert environment["MFQ_MOE_RAM_PCIE_FRACTION"] == "1"
        assert environment["MFQ_MOE_DECODE_HOST_PIN"] == "1"
        assert environment["MFQ_MOE_PREFILL_LAYER"] == "1"
        assert environment["MFQ_MOE_PREFILL_LAYER_ASYNC"] == "1"
        assert environment["MFQ_MOE_RESIDENCY_INTERVAL"] == "4"
        assert float(environment["MFQ_CUDA_VRAM_LIMIT_GIB"]) == 22.5
        assert int(environment["MFQ_MOE_HOST_CACHE_BYTES"]) >= 28 * GIB


def test_cuda_ram_pipeline_rejects_dense_expert_pools(tmp_path):
    path = tmp_path / "dense.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), np.zeros((6, 96), dtype=np.float16)),))
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": experts})
    assert not supports_cuda_ram_expert_pipeline(path)


def test_cuda_ram_pipeline_ignores_resident_dense_banks(tmp_path):
    # A model whose MTP head keeps dense BF16 experts still streams its
    # quantized backbone experts; the dense head runs on the device.
    path = tmp_path / "mixed.mfq"
    experts = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_nvq_fixture()),))
    dense = np.zeros((2, 3, 96), dtype=np.float16)
    save(path, FileHeader(version=2, model_arch="qwen4_exp"), {
        "model.block.0.mlp.experts.gate.weight": experts,
        "predictor.block.1.mlp.experts.up.weight": dense,
    })
    assert supports_cuda_ram_expert_pipeline(path)


def test_cuda_ram_pipeline_needs_a_quantized_expert_bank(tmp_path):
    path = tmp_path / "dense-only.mfq"
    save(path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"model.block.0.mlp.experts.gate.weight": np.zeros((2, 3, 96), dtype=np.float16)})
    assert not supports_cuda_ram_expert_pipeline(path)


def test_cuda_ssd_experts_respect_explicit_ram_cache_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (64 * GIB, 52 * GIB))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 24 * GIB),)))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    pool.runtime_environment["MFQ_MOE_HOST_CACHE_BYTES"] = str(2 * GIB)
    pool.runtime_environment.update(MFQ_MODEL_FILE_IO="buffered", MFQ_PLE_FILE_MAP="0")
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert environment["MFQ_MOE_HOST_CACHE_BYTES"] == str(2 * GIB)
    assert environment["MFQ_MOE_PRELOAD_ALL"] == "0"
    assert environment["MFQ_MODEL_FILE_IO"] == "buffered"
    assert environment["MFQ_PLE_FILE_MAP"] == "0"
    pool.runtime_environment["MFQ_MOE_SSD_EXPERTS"] = "0"
    with pytest.raises(JobExecutionError) as raised:
        pool._launch_configuration(model, request, port=1234)
    assert raised.value.detail.code == "runtime_ram_expert_budget_exceeded"


def _commit_budget_pool(
    tmp_path, monkeypatch, *, architecture, commit_available, available=100 * GIB
):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology",
                        lambda: {"memory_architecture": architecture})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity",
                        lambda: (128 * GIB, available))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_commit_capacity",
                        lambda: (224 * GIB, commit_available))
    monkeypatch.setattr(
        "mfq.server.runtime.runtime_pool.hardware_identity",
        lambda: HardwareIdentity("cpu", 8, gpu_memory=(GpuMemory("RTX GPU", 96 * GIB),)),
    )
    return RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda",
                       automatic_memory_budget=False)


def test_windows_discrete_expert_budget_is_limited_by_commit(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=120 * GIB)
    monkeypatch.setattr(pool, "_cuda_discrete_device_capacity", lambda _request: None)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    plan = pool._cuda_expert_host_budget_plan(model, request)
    # 20 GiB device weights (dense plus cached experts) and 4 GiB workspace
    # are committed from 120 GiB headroom before the reserve, an eighth of
    # the 224 GiB commit limit (28 GiB).
    assert plan.commit_footprint_bytes == 24 * GIB
    assert plan.commit_bytes == 68 * GIB
    assert plan.physical_bytes == 98 * GIB
    assert plan.commit_limited
    assert pool._cuda_expert_host_budget(model, request) == 68 * GIB


def test_windows_commit_footprint_covers_the_worker_vram_limit(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=120 * GIB)
    monkeypatch.setattr(pool, "_cuda_discrete_device_capacity", lambda _request: 96 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    plan = pool._cuda_expert_host_budget_plan(model, request)
    # The worker may grow to its VRAM limit (96 GiB minus 1/16), which WDDM
    # commits even though the load estimate is only 24 GiB.
    assert plan.commit_footprint_bytes == 90 * GIB
    assert plan.commit_bytes == 2 * GIB
    assert pool._cuda_expert_host_budget(model, request) == 2 * GIB


def test_expert_budget_ignores_commit_without_windows_accounting(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=None)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    plan = pool._cuda_expert_host_budget_plan(model, request)
    assert plan.commit_bytes is None and not plan.commit_limited
    assert pool._cuda_expert_host_budget(model, request) == 98 * GIB


def test_unified_expert_budget_keeps_physical_accounting(tmp_path, monkeypatch):
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="unified",
                               commit_available=4 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    # Unified memory already subtracts the estimated load from RAM; commit
    # headroom must not shrink the shared-memory budget a second time.
    assert pool._cuda_expert_host_budget(model, request) == 78 * GIB


def test_windows_commit_pressure_streams_experts_from_ssd(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.delenv("MFQ_MODEL_FILE_IO", raising=False)
    monkeypatch.delenv("MFQ_PLE_FILE_MAP", raising=False)
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=40 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    assert pool._cuda_expert_host_budget(model, request) == 0
    with caplog.at_level(logging.INFO, logger="mfq.server.runtime.runtime_pool"):
        _, environment = pool._launch_configuration(model, request, port=1234)
    assert "limited_by=commit" in caplog.text
    assert environment["MFQ_MOE_PRELOAD_ALL"] == "0"
    assert environment["MFQ_MOE_HOST_CACHE_BYTES"] == "0"
    assert environment["MFQ_MOE_SSD_EXPERTS"] == "1"


def test_preload_requires_room_for_every_routed_expert(tmp_path, monkeypatch):
    monkeypatch.delenv("MFQ_MODEL_FILE_IO", raising=False)
    monkeypatch.delenv("MFQ_PLE_FILE_MAP", raising=False)
    pool = _commit_budget_pool(tmp_path, monkeypatch, architecture="discrete",
                               commit_available=None, available=36 * GIB)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16, device_ids=["0"])
    # Routed experts need 44 GiB and the cold subset only 28 GiB. The 34 GiB
    # host budget fits the cold subset but not a complete preload.
    assert pool._cuda_expert_host_budget(model, request) == 34 * GIB
    _, environment = pool._launch_configuration(model, request, port=1234)
    assert environment["MFQ_MOE_PRELOAD_ALL"] == "0"
    assert environment["MFQ_MOE_HOST_CACHE_BYTES"] == str(34 * GIB)
    assert environment["MFQ_MOE_SSD_EXPERTS"] == "1"


def test_preload_kv_ram_limit_charges_the_complete_expert_set(tmp_path, monkeypatch):
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology",
                        lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity",
                        lambda: (64 * GIB, 52 * GIB))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda",
                       automatic_memory_budget=False)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    assert pool._cuda_expert_host_budget(model, request) == 50 * GIB
    # Preload holds all 44 GiB of routed experts, not only the 28 GiB that
    # the GPU cache cannot mirror, so KV RAM gets 52 - 44 - 2 GiB.
    assert pool._kv_ram_limit(model, request) == 6 * GIB


def test_cuda_expert_cache_leaves_ram_for_preallocated_kv(tmp_path, monkeypatch):
    from mfq.server.runtime.kv_budget import KvTierBudget
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "discrete"})
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (64 * GIB, 12 * GIB))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.checkpoint_kv_budget", lambda *args: KvTierBudget(0, 6 * GIB, 0, 6 * GIB, 0.1))
    pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime", backend="cuda", automatic_memory_budget=False)
    model = artifact(tmp_path)
    request = ModelLoadRequest(model="model", moe_gpu_cache_gb=16)
    assert pool._cuda_expert_host_budget(model, request) == 4 * GIB
    assert pool._kv_ram_limit(model, request) == 6 * GIB


def test_cuda_ram_pipeline_accepts_packed_nint_predictor_experts(tmp_path):
    model = artifact(tmp_path)
    packed = replace(quantize(np.ones((6, 96), dtype=np.float32), NintSpec(4, 24, 6), axis=0), shape=(2, 3, 96))
    save(model.path, FileHeader(version=2, model_arch="qwen4_exp"),
         {"predictor.block.0.mlp.experts.gate_up.weight": packed})
    assert supports_cuda_ram_expert_pipeline(model.path)


def test_cuda_ram_pipeline_accepts_mxfp4_predictor_with_quantized_target(tmp_path):
    from mfq.formats.mx import MxTensor

    model = artifact(tmp_path)
    target = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), make_nvq_fixture()),))
    mx = MxTensor("MXFP4", (6, 96), np.zeros((6, 48), dtype=np.uint8),
                  np.full((6, 3), 127, dtype=np.uint8))
    predictor = MfeTensor((2, 3, 96), (MfePool(np.arange(2, dtype=np.int32), mx),))
    save(model.path, FileHeader(version=2, model_arch="deepseek_v41"), {
        "model.block.0.mlp.experts.gate.weight": target,
        "predictor.stage.0.mlp.experts.down.weight": predictor,
    })
    assert supports_cuda_ram_expert_pipeline(model.path)


@pytest.mark.parametrize("resident_gib,streamed_gib", [(40, 0), (66, 0), (60, 30)])
def test_cuda_pipeline_capable_experts_use_full_cache_when_they_fit(
    tmp_path, monkeypatch, caplog, resident_gib, streamed_gib,
):
    # Decode and MTP verification run as CUDA graphs only through the expert
    # cache, so a pipeline-capable bank caches every expert instead of staying
    # resident on the eager per-expert path.
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=resident_gib + streamed_gib,
        always_streamed_gib=streamed_gib, pipeline=True)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        selected = pool._apply_automatic_expert_residency(model, request)
    assert selected.moe_gpu_cache_gb * GIB > model.routed_expert_bytes
    # Admission still prices the complete checkpoint, not the KV/padding allowance.
    assert pool._estimated_load_bytes(model, selected) == resident_gib * GIB
    command, environment = pool._launch_configuration(model, selected, port=1234)
    assert float(command[command.index("--moe-gpu-cache-gb") + 1]) == selected.moe_gpu_cache_gb
    assert environment["MFQ_MOE_PIPELINE"] == "1" and environment["MFQ_MOE_TWO_STAGE_FFN"] == "1"
    assert float(environment["MFQ_CUDA_VRAM_LIMIT_GIB"]) * GIB == pool._cuda_device_vram_limit(request)
    assert "decision=cache_all" in caplog.text.split()


@pytest.mark.parametrize("context", [4096, 262144])
def test_cuda_cache_all_requests_kv_charge_and_slot_margin(tmp_path, monkeypatch, context):
    # The native cache deducts its GPU KV charge from the requested budget, so
    # a cache that must hold every expert asks for that charge on top.
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    request = ModelLoadRequest(model="model", context_size=context, device_ids=["0"])
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    selected = pool._apply_automatic_expert_residency(model, request)
    margin = max(64 << 20, model.routed_expert_bytes // 100)
    assert selected.moe_gpu_cache_gb * GIB == pytest.approx(model.routed_expert_bytes + reserve.kv_bytes + margin)
    assert pool._estimated_load_bytes(model, selected) == model.resource.total_bytes


@pytest.mark.parametrize("pipeline", [True, False])
@pytest.mark.parametrize("total_gib,cached", [(40, False), (86, False), (100, True)])
def test_cuda_unservable_expert_cache_keeps_previous_residency(
    tmp_path, monkeypatch, caplog, pipeline, total_gib, cached,
):
    # 40 GiB fits with headroom (cache_all candidate), 86 GiB only without the
    # runtime reserve (bounded cache candidate), 100 GiB never fit.
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=total_gib, pipeline=pipeline)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.host_memory_capacity", lambda: (32 * GIB, 8 * GIB))
    if pipeline:
        pool.runtime_environment["MFQ_MOE_SSD_EXPERTS"] = "0"
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        selected = pool._apply_automatic_expert_residency(model, request)
    # A roomy bank the pipeline cannot serve stays resident without a cache candidate.
    if pipeline or total_gib != 40:
        reason = "remaining experts do not fit in RAM" if pipeline else "this expert container requires complete RAM"
        assert f"cache_unavailable='{reason}" in caplog.text
    if not cached:
        # Resident launches never needed host RAM for experts.
        assert selected is request and "decision=resident" in caplog.text.split()
        command, _ = pool._launch_configuration(model, selected, port=1234)
        assert "--moe-gpu-cache-gb" not in command
        return
    # A checkpoint that never fit keeps the bounded cache it requested before.
    assert selected.moe_gpu_cache_gb * GIB == pytest.approx(pool._cuda_device_vram_limit(request) - 4 * GIB)
    with pytest.raises(JobExecutionError) as error:
        pool._launch_configuration(model, selected, port=1234)
    assert error.value.detail.code == "runtime_ram_expert_budget_exceeded"


@pytest.mark.parametrize("architecture,tiered", [
    ("qwen4_exp-hf-mfq-alphaq-ud", True), ("Qwen4_Exp", True), ("qwen3_next", False), ("nemotron_h", False),
])
def test_cuda_residency_reserve_uses_full_kv_outside_qwen4_exp(tmp_path, monkeypatch, architecture, tiered):
    from mfq.server.runtime.kv_budget import checkpoint_kv_budget

    pool, model = _headroom_pool(tmp_path, monkeypatch)
    model = replace(model, resource=model.resource.model_copy(update={"architecture": architecture}))
    request = ModelLoadRequest(model="model", context_size=262144, device_ids=["0"])
    plan = checkpoint_kv_budget(model, 262144, request.kv_memory, request.kv_quantization)
    assert plan.gpu_budget_bytes < plan.indexer_bytes + plan.raw_bytes
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    assert reserve.kv_bytes == (plan.gpu_budget_bytes if tiered else plan.indexer_bytes + plan.raw_bytes)
    explicit = ModelLoadRequest(model="model", context_size=262144, device_ids=["0"],
        kv_memory={"gpu_budget_bytes": 8 * GIB})
    assert pool._cuda_residency_runtime_reserve(model, explicit).kv_bytes == 8 * GIB


def _save_config(path, config):
    save(path, FileHeader(version=2, model_arch=str(config.get("model_type", "unknown")),
        extra={"hf_config": config}), {"weight": np.zeros((2, 2), dtype=np.float16)})


@pytest.mark.parametrize("config,expected", [
    # Nemotron-H block list (legacy mixer names): 4 attention layers.
    (dict(model_type="nemotron_h", num_hidden_layers=99, num_key_value_heads=2, head_dim=128,
          layers_block_type=["mamba", "moe", "attention", "moe"] * 4), 4 * 2 * 128 * 4),
    # Omni wrapper nests the language model under llm_config.
    (dict(model_type="nemotron_h_omni", llm_config=dict(num_key_value_heads=2, head_dim=128,
          layers_block_type=["linear_attention", "full_attention"] * 3)), 3 * 2 * 128 * 4),
    (dict(num_hidden_layers=8, num_key_value_heads=4, head_dim=64, hybrid_override_pattern="M-M*M-M*"),
     2 * 4 * 64 * 4),
    (dict(num_hidden_layers=48, full_attention_interval=4, num_key_value_heads=2, head_dim=256), 12 * 2 * 256 * 4),
    # head_dim derived from hidden_size and KV heads from attention heads.
    (dict(num_hidden_layers=10, hidden_size=1024, num_attention_heads=8), 10 * 8 * 128 * 4),
    (dict(layers_block_type=["mamba", "moe"] * 4), 0),
    (dict(num_hidden_layers=10), None),
    ({}, None),
])
def test_attention_kv_estimate_reads_checkpoint_config(tmp_path, config, expected):
    from mfq.server.services.model_memory import checkpoint_attention_kv_bytes_per_token

    model = artifact(tmp_path)
    _save_config(model.path, config)
    assert checkpoint_attention_kv_bytes_per_token(model) == expected


def test_cuda_residency_fallback_derives_kv_from_config(tmp_path, monkeypatch):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    monkeypatch.setattr("mfq.server.services.model_memory.checkpoint_cache_profile", lambda *args, **kwargs: None)
    _save_config(model.path, dict(model_type="nemotron_h", num_key_value_heads=2, head_dim=128,
        layers_block_type=["mamba", "moe", "attention", "moe"] * 4))
    request = ModelLoadRequest(model="model", context_size=1048576, device_ids=["0"])
    reserve = pool._cuda_residency_runtime_reserve(model, request)
    assert reserve.kv_source == "config_attention_fp16_min_1gib"
    assert reserve.kv_bytes == 1048576 * 4 * 2 * 128 * 4
    short = pool._cuda_residency_runtime_reserve(model, request.model_copy(update={"context_size": 4096}))
    assert short.kv_bytes == GIB


def test_cuda_residency_fallback_without_geometry_is_capped_at_ceiling(tmp_path, monkeypatch, caplog):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    monkeypatch.setattr("mfq.server.services.model_memory.checkpoint_cache_profile", lambda *args, **kwargs: None)
    request = ModelLoadRequest(model="model", device_ids=["0"])
    ceiling = pool._cuda_device_vram_limit(request)
    assert pool.resolve_context_size(model, None) == 1048576
    # 256 KiB/token at the 1M default context would claim 256 GiB.
    assert pool._cuda_residency_runtime_reserve(model, request).kv_bytes == 256 * GIB
    assert pool._cuda_residency_runtime_reserve(model, request, ceiling=ceiling).kv_bytes == ceiling
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        selected = pool._apply_automatic_expert_residency(model, request)
    assert f"kv_reserve_bytes={ceiling}" in caplog.text.split()
    assert "decision=cache" in caplog.text.split() and selected.moe_gpu_cache_gb > 0


def test_cuda_unified_memory_keeps_resident_experts(tmp_path, monkeypatch, caplog):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.hardware_identity", lambda: HardwareIdentity(
        "cpu", 8, gpu_memory=(GpuMemory("NVIDIA GB10", 128 * GIB, unified=True),)))
    monkeypatch.setattr("mfq.server.runtime.runtime_pool.memory_topology", lambda: {"memory_architecture": "unified"})
    assert supports_cuda_ram_expert_pipeline(model.path)
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    with caplog.at_level("INFO", logger="mfq.server.runtime.runtime_pool"):
        assert pool._apply_automatic_expert_residency(model, request, memory_ceiling=100 * GIB) is request
    assert "decision=resident" in caplog.text.split()


@pytest.mark.parametrize("inherited", [False, True])
def test_cuda_disabled_pipeline_keeps_resident_experts(tmp_path, monkeypatch, inherited):
    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    if inherited:
        monkeypatch.setenv("MFQ_MOE_PIPELINE", "0")
    else:
        pool.runtime_environment["MFQ_MOE_PIPELINE"] = "0"
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    assert pool._apply_automatic_expert_residency(model, request) is request


def test_cuda_residency_and_launch_probe_checkpoint_once(tmp_path, monkeypatch):
    import mfq.server.runtime.native as native
    import mfq.server.services.model_memory as model_memory

    pool, model = _headroom_pool(tmp_path, monkeypatch, total_gib=40)
    # Real metadata path: the fixture has no KV geometry, so the reserve also
    # falls back to the config estimate after the profile lookup.
    monkeypatch.setattr("mfq.server.services.model_memory.checkpoint_cache_profile", checkpoint_cache_profile)
    probes, reads = [], []
    inspect, open_store = native.inspect_mfe_header, model_memory.open_mmap
    monkeypatch.setattr(native, "inspect_mfe_header", lambda *args: probes.append(1) or inspect(*args))
    monkeypatch.setattr(model_memory, "open_mmap",
        lambda *args, **kwargs: reads.append(1) or open_store(*args, **kwargs))
    request = ModelLoadRequest(model="model", context_size=4096, device_ids=["0"])
    selected = pool._apply_automatic_expert_residency(model, request)
    assert selected.moe_gpu_cache_gb > 0
    pool._launch_configuration(model, selected, port=1234)
    assert supports_cuda_ram_expert_pipeline(model.path)
    assert len(probes) == 1 and len(reads) == 1
    # A rewritten checkpoint is probed again.
    stat = model.path.stat()
    os.utime(model.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert supports_cuda_ram_expert_pipeline(model.path)
    assert len(probes) == 2
