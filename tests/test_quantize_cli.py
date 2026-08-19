from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mfq import cli


def _hf_source(tmp_path: Path) -> Path:
    source = tmp_path / "hf-model"
    source.mkdir()
    (source / "model-00001-of-00001.safetensors").write_bytes(b"")
    return source


def test_quantize_routes_hf_common_options(tmp_path: Path, monkeypatch) -> None:
    source = _hf_source(tmp_path)
    output = tmp_path / "model.mfq"
    recipe = tmp_path / "recipe.gguf"
    recipe.write_bytes(b"GGUF")
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_hf_to_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(output),
                "--recipe",
                str(recipe),
                "--scheme",
                str(tmp_path / "ew.json"),
                "--bits",
                "5",
                "--split-max-size",
                "2G",
                "--resume",
                "--dry-run",
            ]
        )
        == 0
    )
    assert len(captured) == 1
    args = captured[0]
    assert args.input == str(source.resolve())
    assert args.output == str(output.resolve())
    assert args.recipe_gguf == str(recipe)
    assert args.calibration_scheme == str(tmp_path / "ew.json")
    assert args.bits == 5
    assert args.split_max_size == 2_000_000_000
    assert args.resume_temp is True
    assert args.dry_run is True


def test_quantize_routes_metal_backend(tmp_path: Path, monkeypatch) -> None:
    source = _hf_source(tmp_path)
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_hf_to_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "model.mfq"),
                "--backend",
                "metal",
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured[0].quant_backend == "metal"


def test_quantize_routes_gguf_imatrix_and_q8_mode(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "bf16.gguf"
    recipe = tmp_path / "recipe.gguf"
    source.write_bytes(b"GGUF")
    recipe.write_bytes(b"GGUF")
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_gguf_to_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "model.mfq"),
                "--recipe",
                str(recipe),
                "--imatrix",
                str(tmp_path / "imatrix.gguf"),
                "--ew-scheme",
                str(tmp_path / "ew.json"),
                "--q8-mode",
                "nint8-0",
                "--resume-completed",
                "7",
                "--dry-run",
            ]
        )
        == 0
    )
    args = captured[0]
    assert args.input_bf16_gguf == str(source.resolve())
    assert args.recipe_gguf == str(recipe.resolve())
    assert args.imatrix == str(tmp_path / "imatrix.gguf")
    assert args.calibration_scheme == str(tmp_path / "ew.json")
    assert args.q8_to_nint8_zero is True
    assert args.resume_completed == 7


def test_quantize_in_uses_existing_baseline(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "bf16.gguf"
    recipe = tmp_path / "recipe.gguf"
    baseline = tmp_path / "base.mfq"
    for path in (source, recipe, baseline):
        path.write_bytes(b"data")
    base_calls: list[argparse.Namespace] = []
    in_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_gguf_to_mfq.convert", base_calls.append
    )
    monkeypatch.setattr(
        "mfq.tools.quantize_important_neurons.convert", in_calls.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "in.mfq"),
                "--recipe",
                str(recipe),
                "--imatrix",
                str(tmp_path / "imatrix.gguf"),
                "--important-neurons",
                "1024",
                "--in-baseline",
                str(baseline),
                "--in-layers",
                "64",
                "--target-size",
                "15G",
                "--dry-run",
            ]
        )
        == 0
    )
    assert base_calls == []
    assert len(in_calls) == 1
    args = in_calls[0]
    assert args.baseline_mfq == str(baseline.resolve())
    assert args.top_k == 1024
    assert args.layers == 64
    assert args.target_bytes == 15_000_000_000


def test_quantize_rejects_gguf_without_recipe(tmp_path: Path) -> None:
    source = tmp_path / "bf16.gguf"
    source.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="requires --recipe"):
        cli.main(["quantize", str(source), str(tmp_path / "model.mfq")])


def test_quantize_routes_hf_imatrix(tmp_path: Path, monkeypatch) -> None:
    source = _hf_source(tmp_path)
    imatrix = tmp_path / "imatrix.gguf"
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_hf_to_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "model.mfq"),
                "--imatrix",
                str(imatrix),
            ]
        )
        == 0
    )
    assert len(captured) == 1
    assert captured[0].imatrix == str(imatrix)


def test_quantize_routes_hf_vq_options_and_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _hf_source(tmp_path)
    overrides = tmp_path / "overrides.json"
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_hf_to_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "model.mfq"),
                "--tensor-overrides",
                str(overrides),
                "--nvq-codebook-scope",
                "fixed",
                "--nvq-calibration",
                "none",
                "--nvq3-jsc-512",
                "--q8-mode",
                "nint8-0",
            ]
        )
        == 0
    )
    assert len(captured) == 1
    args = captured[0]
    assert args.tensor_precision_overrides == str(overrides)
    assert args.nvq_codebook_scope == "fixed"
    assert args.nvq_calibration == "none"
    assert args.nvq3_jsc_512 is True
    assert args.q8_to_nint8_zero is True


def test_quantize_routes_hf_bf16(tmp_path: Path, monkeypatch) -> None:
    source = _hf_source(tmp_path)
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.convert_hf_to_full_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "model.mfq"),
                "--full-precision",
            ]
        )
        == 0
    )

    assert len(captured) == 1
    assert captured[0].input == str(source.resolve())


def test_quantize_auto_detects_and_routes_full_precision_mfq(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "full.mfq"
    source.write_bytes(b"MFQ1")
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        "mfq.tools.quantize_hf_to_mfq.convert", captured.append
    )

    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "low.mfq"),
                "--bits",
                "3",
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured[0].input_mfq == str(source.resolve())
    assert captured[0].bits == 3


def test_quantize_rejects_bf16_with_quantization_recipe(tmp_path: Path) -> None:
    source = _hf_source(tmp_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        cli.main(
            [
                "quantize",
                str(source),
                str(tmp_path / "model.mfq"),
                "--bf16",
                "--recipe",
                str(tmp_path / "recipe.gguf"),
            ]
        )
