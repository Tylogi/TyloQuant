from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.calibration.artifact import ExpertPrecision
from mfq.cli import _build_parser
from mfq.formats.tpq import CCCP_X, pack_cccp_pq
from mfq.formats.header import FileHeader
from mfq.formats.io import load_mmap, save
from mfq.formats.shards import write_blob_record_shards
from mfq.formats.tpq import (
    TPQ_X,
    TpqPqTensor,
    normalize_tpq_dtype,
    pack_tpq_pq,
)
from mfq.quantize.tpq import quantize_tpq_pq_fixed
from mfq.runtime.tpq import TPQArtifact


@dataclass(frozen=True)
class _BlobRecord:
    name: str
    dtype: str
    path: Path
    nbytes: int
    offset: int = 0


def _tpq_tensor() -> TpqPqTensor:
    rng = np.random.default_rng(20260731)
    weight = rng.normal(size=(3, 24)).astype(np.float32)
    codebook = rng.normal(
        size=(TPQ_X.codebook_entries, TPQ_X.vector_size)
    ).astype(np.float32)
    return quantize_tpq_pq_fixed(
        weight,
        TPQ_X,
        codebook,
        device="cpu",
    )


def test_new_writer_uses_tpq_dtype(tmp_path: Path) -> None:
    output = tmp_path / "new.mfq"
    save(
        output,
        FileHeader(version=2, model_arch="test", num_tensors=1),
        {"weight": _tpq_tensor()},
    )
    _, store = load_mmap(output)
    try:
        assert store.records["weight"].dtype == "TPQ-X"
        assert store["weight"].spec.label == "TPQ-X"
    finally:
        store.close()


def test_tpq_rename_preserves_pq_payload_bytes() -> None:
    tensor = _tpq_tensor()
    payload = pack_tpq_pq(tensor)
    assert CCCP_X is TPQ_X
    assert pack_cccp_pq(tensor) == payload
    assert payload.startswith(b"CPQ1")


def test_reader_accepts_legacy_cccp_dtype(tmp_path: Path) -> None:
    payload = pack_tpq_pq(_tpq_tensor())
    blob = tmp_path / "weight.blob"
    blob.write_bytes(payload)
    output = tmp_path / "legacy.mfq"
    write_blob_record_shards(
        output,
        FileHeader(version=2, model_arch="test", num_tensors=1),
        [_BlobRecord("weight", "CCCP-X", blob, len(payload))],
    )
    _, store = load_mmap(output)
    try:
        assert store.records["weight"].dtype == "CCCP-X"
        assert store["weight"].spec.label == "TPQ-X"
    finally:
        store.close()


def test_legacy_scheme_family_normalizes_to_tpq() -> None:
    assert ExpertPrecision(family="CCCP-V").family == "TPQ-V"


def test_legacy_projection_vq_dtype_normalizes_to_tpq_p() -> None:
    assert normalize_tpq_dtype("TPQ-PVQ") == "TPQ-P"
    assert normalize_tpq_dtype("CCCP-PVQ") == "TPQ-P"


def test_cli_exposes_tpq_and_legacy_cccp_groups() -> None:
    parser = _build_parser()
    assert parser.parse_args(["tpq", "inspect", "model"]).command == "tpq"
    assert parser.parse_args(["cccp", "inspect", "model"]).command == "cccp"


def test_legacy_cccp_directory_remains_readable(tmp_path: Path) -> None:
    root = tmp_path / "cccp"
    root.mkdir()
    dense = root / "dense.safetensors"
    dense.write_bytes(b"dense")
    expert = root / "experts.safetensors"
    expert.write_bytes(b"expert")
    manifest = {
        "format": "cccp-1",
        "config": {
            "n_layers": 1,
            "n_experts": 1,
            "top_k": 1,
            "hidden": 8,
            "moe_inter": 8,
            "hc_mult": 1,
        },
        "quant": {"vq": {"x": [8, 256]}},
        "expert_files": {"0": expert.name},
        "dense_file": dense.name,
        "tokenizer_files": [],
    }
    (root / "cccp.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    artifact = TPQArtifact.open(root)
    assert artifact.manifest["format"] == "cccp-1"


def test_canonical_tpq_directory_is_preferred(tmp_path: Path) -> None:
    root = tmp_path / "tpq"
    root.mkdir()
    (root / "dense.safetensors").write_bytes(b"dense")
    (root / "experts.safetensors").write_bytes(b"expert")
    manifest = {
        "format": "tpq-1",
        "config": {
            "n_layers": 1,
            "n_experts": 1,
            "top_k": 1,
            "hidden": 8,
            "moe_inter": 8,
            "hc_mult": 1,
        },
        "quant": {"vq": {"x": [8, 256]}},
        "expert_files": {"0": "experts.safetensors"},
        "dense_file": "dense.safetensors",
        "tokenizer_files": [],
    }
    (root / "tpq.json").write_text(json.dumps(manifest), encoding="utf-8")
    legacy = dict(manifest, format="cccp-1")
    (root / "cccp.json").write_text(json.dumps(legacy), encoding="utf-8")
    artifact = TPQArtifact.open(root)
    assert artifact.manifest["format"] == "tpq-1"
