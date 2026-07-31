from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mfq.runtime.tpq import (
    TPQArtifact,
    _native_tokenizer_host,
    _pop_argument,
    load_tpq_package,
)
from mfq.runtime.tpq_residency_patch import apply_tpq_residency_patch


def _artifact(root: Path, *, layers: int = 2) -> Path:
    root.mkdir()
    files = {
        "cccp.json": b"",
        "dense.safetensors": b"dense",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "generation_config.json": b"{}",
        "dspark-vq.safetensors": b"dspark",
    }
    expert_files = {}
    for layer in range(layers):
        name = f"experts.L{layer:02d}.safetensors"
        files[name] = f"expert-{layer}".encode()
        expert_files[str(layer)] = name
    manifest = {
        "format": "cccp-1",
        "config": {
            "n_layers": layers,
            "n_experts": 4,
            "top_k": 2,
            "hc_mult": 4,
        },
        "quant": {
            "dense": "int4-g64",
            "int4_group": 64,
            "vq": {"v": [4, 256], "w": [4, 80]},
        },
        "tiers_per_layer": {"0": "vvww", "1": "vwvw"},
        "dense_file": "dense.safetensors",
        "expert_files": expert_files,
        "tokenizer_files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ],
        "dspark_file": "dspark-vq.safetensors",
        "dspark": {"n_layers": 3, "targets": [40, 41, 42], "k": 5},
    }
    files["cccp.json"] = json.dumps(manifest).encode()
    for name, value in files.items():
        (root / name).write_bytes(value)
    return root


def test_cccp_artifact_validates_and_summarizes(tmp_path: Path) -> None:
    artifact = TPQArtifact.open(_artifact(tmp_path / "model"))
    summary = artifact.summary()

    assert artifact.architecture == "deepseek_v4"
    assert summary["format"] == "cccp-1"
    assert summary["files"] == 8
    assert summary["expert_tier_counts"] == {"v": 4, "w": 4}
    assert summary["vq"] == {"v": [4, 256], "w": [4, 80]}


def test_cccp_artifact_rejects_missing_shard(tmp_path: Path) -> None:
    root = _artifact(tmp_path / "model")
    (root / "experts.L01.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="incomplete"):
        TPQArtifact.open(root)


def test_cccp_artifact_recognizes_kimi_dense_shards_and_sparse_moe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "kimi"
    dense = root / "dense"
    dense.mkdir(parents=True)
    (dense / "model-00001.safetensors").write_bytes(b"dense-1")
    (dense / "model-00002.safetensors").write_bytes(b"dense-2")
    expert_files = {}
    for layer in (2, 3):
        name = f"experts.L{layer:02d}.safetensors"
        (root / name).write_bytes(f"expert-{layer}".encode())
        expert_files[str(layer)] = name
    manifest = {
        "format": "cccp-1",
        "model_family": "kimi_k3",
        "config": {
            "n_layers": 4,
            "first_dense_layers": 2,
            "routed_hidden": 64,
            "kda_layers": [0, 2],
            "n_experts": 4,
            "top_k": 2,
        },
        "quant": {
            "dense": "int4-g64",
            "int4_group": 64,
            "vq": {"x": [8, 256]},
        },
        "dense_files": [
            "model-00001.safetensors",
            "model-00002.safetensors",
        ],
        "nonexpert": {"path": "dense"},
        "expert_files": expert_files,
        "tiers_per_layer": {"2": "xxxd", "3": "xxxx"},
    }
    (root / "cccp.json").write_text(json.dumps(manifest), encoding="utf-8")

    artifact = TPQArtifact.open(root)
    assert artifact.architecture == "kimi_k3"
    assert artifact.expert_bytes == sum(
        (root / name).stat().st_size for name in expert_files.values()
    )
    assert len(artifact.files) == 5


def test_checked_in_tpq_package_can_be_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MFQ_TPQ_ROOT", raising=False)
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "tpq" or name.startswith("tpq.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        package = load_tpq_package()
        assert package.__name__ == "tpq"
        assert Path(package.__file__).name == "__init__.py"
        assert Path(package.__file__).parent.name == "tpq"
        assert Path(package.__file__).parent.parent.name == "_vendor"
    finally:
        for name in list(sys.modules):
            if name == "tpq" or name.startswith("tpq."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_explicit_tpq_root_takes_precedence(tmp_path: Path) -> None:
    root = tmp_path / "tpq-explicit"
    root.mkdir()
    (root / "__init__.py").write_text("source = 'explicit'\n", encoding="utf-8")
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "tpq" or name.startswith("tpq.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        package = load_tpq_package(root)
        assert package.source == "explicit"
        assert Path(package.__file__).parent == root
    finally:
        for name in list(sys.modules):
            if name == "tpq" or name.startswith("tpq."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_vendored_tpq_exposes_kimi_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPQ_FUSED", "0")
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "tpq" or name.startswith("tpq.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        load_tpq_package()
        engine = __import__("tpq.engine", fromlist=["_make_model"])
        kimi_model = __import__("tpq.kimi_model", fromlist=["KimiK3TPQModel"])
        check = __import__("tpq.check", fromlist=["_self_test"])

        assert "tp_size" in engine._make_model.__annotations__
        assert kimi_model.KimiK3TPQModel.__name__ == "KimiK3TPQModel"
        check._self_test()
    finally:
        for name in list(sys.modules):
            if name == "tpq" or name.startswith("tpq."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_invalid_explicit_tpq_root_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="entry point not found"):
        load_tpq_package(tmp_path / "missing")


def test_tpq_residency_patch_is_idempotent() -> None:
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "tpq" or name.startswith("tpq.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        package = load_tpq_package()
        apply_tpq_residency_patch()
        apply_tpq_residency_patch()

        assert package.store.ExpertPool.preload_gpu_all
        assert package.store.ExpertPool.preload_all._mfq_cgroup_resident
        assert package.store.CCCPStore.expert_signature_counts
    finally:
        for name in list(sys.modules):
            if name == "tpq" or name.startswith("tpq."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_tpq_host_residency_accepts_native_store_without_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "tpq" or name.startswith("tpq.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        package = load_tpq_package()
        apply_tpq_residency_patch()
        weight = package.store.VQWeight(
            torch.zeros((1, 1), dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.float32),
            1,
        )
        signature = type("Signature", (), {"slot_bytes": 2})()
        native_store = SimpleNamespace(
            cfg={"n_experts": 1},
            man=SimpleNamespace(expert_files={0: "native-record"}),
            expert_signature_counts=lambda: {signature: 1},
            expert_kind=lambda _layer, _expert: "v",
            load_expert=lambda _layer, _expert: (weight, weight),
            drop_dense_file_cache=lambda: None,
            drop_expert_file_cache=lambda _layer: None,
        )
        pool = package.store.ExpertPool.__new__(package.store.ExpertPool)
        pool.store = native_store
        pool.gpu = False
        pool.pinned = {}
        monkeypatch.setenv("TPQ_FULL_RESIDENT", "1")
        monkeypatch.setenv("TPQ_HOST_PIN_GB", "0")

        assert pool.preload_all(reserve_gb=0.0)
        assert list(pool.pinned) == [(0, 0)]
    finally:
        for name in list(sys.modules):
            if name == "tpq" or name.startswith("tpq."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_native_tokenizer_host_materializes_engine_inputs(
    tmp_path: Path,
) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer_root / "generation_config.json").write_text(
        '{"eos_token_id":1}',
        encoding="utf-8",
    )
    artifact = SimpleNamespace(
        manifest={
            "format": "cccp-1",
            "config": {"hc_mult": 4},
            "tokenizer_files": [
                "tokenizer.json",
                "generation_config.json",
            ],
        }
    )
    with _native_tokenizer_host(artifact, tokenizer_root) as host:
        assert json.loads((host / "cccp.json").read_text())["format"] == "cccp-1"
        assert (host / "tokenizer.json").read_text() == "{}"
        assert (host / "generation_config.json").is_file()


def test_pop_argument_removes_native_only_option() -> None:
    arguments = ["--model", "model.mfq", "--tokenizer-root", "tokenizer"]
    assert _pop_argument(arguments, "--tokenizer-root") == "tokenizer"
    assert arguments == ["--model", "model.mfq"]
