from __future__ import annotations

import json

import numpy as np

from mfq.formats.nvq import E8_256
from mfq.quantize.nvq_codebook import (
    NvqCodebookArtifact,
    NvqCodebookTrainingConfig,
    NvqTrainingMatrix,
    TrainedNvqCodebook,
    codebook_scope_key,
    load_nvq_codebook_artifact,
    pack_e8_codebook,
    save_nvq_codebook_artifact,
    train_nvq2_codebook,
    unpack_e8_codebook,
)


def test_e8_codebook_packs_to_512_bytes():
    payload = pack_e8_codebook(E8_256)
    assert len(payload) == 512
    np.testing.assert_array_equal(unpack_e8_codebook(payload), E8_256)


def test_scope_keys_cover_model_family_and_tensor():
    name = "model.layers.3.mlp.down_proj.weight"
    assert codebook_scope_key(name, "model") == "model"
    assert codebook_scope_key(name, "family") == "ffn_down"
    assert codebook_scope_key(name, "tensor") == name


def test_codebook_artifact_roundtrip(tmp_path):
    config = NvqCodebookTrainingConfig(iterations=0)
    trained = TrainedNvqCodebook(
        key="model",
        tensor_names=("a.weight",),
        table=E8_256,
        history=({"iteration": 0, "nmse_percent": 1.0},),
        train_elements=128,
        train_signal=2.5,
    )
    artifact = NvqCodebookArtifact("synthetic", config, (trained,), {"a.weight": [1, 3]})
    path = save_nvq_codebook_artifact(artifact, tmp_path / "codebook.json")
    restored = load_nvq_codebook_artifact(path)
    assert restored.config == config
    assert restored.source_rows == {"a.weight": [1, 3]}
    np.testing.assert_array_equal(restored.codebooks[0].table, E8_256)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["format"] = "mfq.niq2.codebooks"
    legacy_path = tmp_path / "legacy-codebook.json"
    legacy_path.write_text(json.dumps(document), encoding="utf-8")
    legacy = load_nvq_codebook_artifact(legacy_path)
    np.testing.assert_array_equal(legacy.codebooks[0].table, E8_256)


def test_one_lloyd_iteration_returns_legal_unique_grid():
    rng = np.random.default_rng(15)
    matrix = NvqTrainingMatrix(
        "model.layers.0.mlp.gate_proj.weight",
        rng.normal(0, 0.04, size=(2, 48)).astype(np.float32),
    )
    config = NvqCodebookTrainingConfig(
        iterations=1,
        search_steps=2,
        scale_refine_steps=1,
        group_chunk=8,
        projection_candidates=3,
        reseed_pool_size=16,
    )
    table, history, elements, signal = train_nvq2_codebook((matrix,), config)
    assert table.shape == (256, 8)
    assert np.unique(table, axis=0).shape[0] == 256
    assert set(np.unique(table)).issubset({1, 3, 5, 7})
    assert len(history) == 2
    assert elements == matrix.weight.size
    assert signal > 0
