from __future__ import annotations

import json

import pytest

from mfq.formats.runtime_profile import (
    RUNTIME_SAMPLING_METADATA_KEY,
    architecture_profile,
    load_mfq_sidecar,
    merge_runtime_profiles,
    model_profile,
    profile_for_new_mfq,
    sidecar_paths,
    validate_runtime_profile,
)


def test_architecture_registry_is_partial() -> None:
    profile = architecture_profile("MiniCPMOForCausalLM")
    assert profile is not None
    assert profile["chat"]["temperature"] == 0.7
    assert profile["chat"]["enable_thinking"] is False
    assert profile["duplex"]["force_listen_count"] == 0
    assert profile["duplex"]["system_prompt"] == "Streaming Omni Conversation."
    assert "max_tokens" not in profile["chat"]


def test_exact_model_registry_matches_repository_identity() -> None:
    profile = model_profile("Tylogi/MiniCPM-o-4_5-MFQ")
    assert profile is not None
    assert profile["provenance"]["source"] == "model-registry:minicpm-o-4_5"


def test_unknown_architecture_does_not_invent_metadata(tmp_path) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    assert profile_for_new_mfq(source, {"model_type": "unknown_new_model"}) is None


def test_generation_config_overrides_architecture_registry(tmp_path) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "generation_config.json").write_text(
        json.dumps({"temperature": 0.33, "max_new_tokens": 777}),
        encoding="utf-8",
    )
    profile = profile_for_new_mfq(source, {"model_type": "minicpmo"})
    assert profile is not None
    assert profile["chat"]["temperature"] == 0.33
    assert profile["chat"]["max_tokens"] == 777
    assert profile["chat"]["top_p"] == 0.8


def test_merge_is_fieldwise_and_later_profile_wins() -> None:
    result = merge_runtime_profiles(
        {"chat": {"temperature": 0.7, "top_p": 0.8}},
        {"chat": {"temperature": 0.2}},
    )
    assert result is not None
    assert result["chat"] == {"temperature": 0.2, "top_p": 0.8}


def test_sharded_sidecar_family_then_exact_priority(tmp_path) -> None:
    shard = tmp_path / "model-00002-of-00003.mfq"
    family, exact = sidecar_paths(shard)
    family.write_text(
        json.dumps({"chat": {"temperature": 0.4, "top_p": 0.7}}),
        encoding="utf-8",
    )
    exact.write_text(
        json.dumps({"chat": {"temperature": 0.5}}),
        encoding="utf-8",
    )
    profile = load_mfq_sidecar(shard)
    assert profile is not None
    assert profile["chat"] == {"temperature": 0.5, "top_p": 0.7}


def test_invalid_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_runtime_profile({"chat": {"temperature": float("nan")}})
    with pytest.raises(ValueError, match="version"):
        validate_runtime_profile({"version": 2, "chat": {"top_p": 0.8}})
    with pytest.raises(ValueError, match="boolean"):
        validate_runtime_profile({"chat": {"enable_thinking": 0}})


def test_metadata_key_is_versioned() -> None:
    assert RUNTIME_SAMPLING_METADATA_KEY == "runtime.sampling.v1"
