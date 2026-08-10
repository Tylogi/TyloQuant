from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import torch

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.header import FileHeader
from mfq.formats.io import BFloat16Array, open_mmap, save
from mfq.formats.mx import MxTensor
from mfq.formats.nint import NintSpec
from mfq.formats.runtime_profile import RUNTIME_SAMPLING_METADATA_KEY
from mfq.quantize.mfq_source import FullPrecisionMfqCheckpoint
from mfq.quantize.nint_quant import quantize
from mfq.tools.convert_hf_to_full_mfq import build_parser as build_full_parser
from mfq.tools.convert_hf_to_full_mfq import convert as convert_full
from mfq.tools.quantize_hf_to_mfq import build_parser as build_quantize_parser
from mfq.tools.quantize_hf_to_mfq import convert as quantize_full


def _write_safetensor(path, tensors):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(bytes(data))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _bf16(values: torch.Tensor) -> BFloat16Array:
    return (
        values.to(torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .view(BFloat16Array)
    )


def _full_sample(path):
    save(
        path,
        FileHeader(
            version=2,
            model_arch="deepseek_v4-hf-full-mfq",
            extra={"full_precision_mfq": True},
        ),
        {
            "norm.weight": _bf16(torch.arange(8, dtype=torch.float32)),
            "fp8.weight": MxTensor(
                "MXFP8",
                (128, 128),
                np.full((128, 128), 0x38, dtype=np.uint8),
                np.full((1, 1), 127, dtype=np.uint8),
            ),
            "fp4.weight": MxTensor(
                "MXFP4",
                (4, 96),
                np.arange(4 * 48, dtype=np.uint8).reshape(4, 48),
                np.full((4, 3), 127, dtype=np.uint8),
            ),
            MODEL_CONFIG_ASSET: json.dumps(
                {"model_type": "deepseek_v4"}
            ).encode(),
        },
    )


def test_native_hf_dtypes_convert_to_self_contained_full_precision_mfq(tmp_path):
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
    )
    bf16 = np.arange(8, dtype=np.uint16)
    fp8 = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
    fp8_scale = np.full((1, 1), 127, dtype=np.uint8)
    fp4 = np.arange(4 * 48, dtype=np.uint8).reshape(4, 48)
    fp4_scale = np.full((4, 3), 127, dtype=np.uint8)
    _write_safetensor(
        root / "model-00001-of-00001.safetensors",
        {
            "norm.weight": ("BF16", (8,), bf16.tobytes()),
            "fp8.scale": ("F8_E8M0", fp8_scale.shape, fp8_scale.tobytes()),
            "fp8.weight": ("F8_E4M3", fp8.shape, fp8.tobytes()),
            "fp4.scale": ("F8_E8M0", fp4_scale.shape, fp4_scale.tobytes()),
            "fp4.weight": ("I8", fp4.shape, fp4.tobytes()),
        },
    )
    output = tmp_path / "full.mfq"
    convert_full(
        build_full_parser().parse_args(
            ["--input", str(root), "--output", str(output)]
        )
    )

    with FullPrecisionMfqCheckpoint(output) as checkpoint:
        assert checkpoint.info("norm.weight").dtype == "BF16"
        assert checkpoint.info("fp8.weight").dtype == "MXFP8"
        assert checkpoint.info("fp8.weight").shape == (128, 128)
        assert checkpoint.info("fp4.weight").dtype == "MXFP4"
        assert checkpoint.info("fp4.weight").shape == (4, 96)
        assert "fp8.scale" not in checkpoint.store.records
        assert "fp4.scale" not in checkpoint.store.records
        profile = checkpoint.header.extra[RUNTIME_SAMPLING_METADATA_KEY]
        assert profile["chat"]["top_p"] == 0.8
        assert "max_tokens" not in profile["chat"]


def test_full_precision_mfq_quantizes_bf16_fp8_and_mxfp4(tmp_path):
    source = tmp_path / "full.mfq"
    output = tmp_path / "nint3.mfq"
    _full_sample(source)
    quantize_full(
        build_quantize_parser().parse_args(
            [
                "--input-mfq",
                str(source),
                "--output",
                str(output),
                "--bits",
                "3",
                "--groupsize",
                "24",
                "--sub-bits",
                "5",
                "--row-chunk",
                "4",
                "--quant-backend",
                "cpu",
                "--device",
                "cpu",
            ]
        )
    )

    with open_mmap(output) as store:
        assert store.records["norm.weight"].dtype == "F32"
        assert store.records["fp8.weight"].dtype == "NINT3"
        assert store.records["fp4.weight"].dtype == "NINT3"
        assert MODEL_CONFIG_ASSET in store.records
        assert store.header.extra["source_format"] == "mfq"
        assert (
            store.header.extra[RUNTIME_SAMPLING_METADATA_KEY]["chat"]["top_p"]
            == 0.8
        )


def test_mxfp4_cannot_be_requantized_to_equal_or_higher_precision(tmp_path):
    source = tmp_path / "full.mfq"
    _full_sample(source)
    args = build_quantize_parser().parse_args(
        [
            "--input-mfq",
            str(source),
            "--output",
            str(tmp_path / "bad.mfq"),
            "--bits",
            "4",
            "--quant-backend",
            "cpu",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="MXFP4.*NINT4"):
        quantize_full(args)


def test_mxfp8_cannot_be_requantized_to_equal_precision(tmp_path):
    source = tmp_path / "full.mfq"
    save(
        source,
        FileHeader(version=2, model_arch="fp8-full-mfq"),
        {
            "fp8.weight": MxTensor(
                "MXFP8",
                (128, 128),
                np.full((128, 128), 0x38, dtype=np.uint8),
                np.full((1, 1), 127, dtype=np.uint8),
            )
        },
    )
    args = build_quantize_parser().parse_args(
        [
            "--input-mfq",
            str(source),
            "--output",
            str(tmp_path / "bad.mfq"),
            "--bits",
            "8",
            "--quant-backend",
            "cpu",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="MXFP8.*NINT8"):
        quantize_full(args)


def test_full_precision_mfq_uses_the_shared_vq_writer(tmp_path):
    source = tmp_path / "full.mfq"
    output = tmp_path / "nvq.mfq"
    overrides = tmp_path / "overrides.json"
    _full_sample(source)
    overrides.write_text(
        json.dumps({"fp4.weight": "NVQ2"}), encoding="utf-8"
    )
    quantize_full(
        build_quantize_parser().parse_args(
            [
                "--input-mfq",
                str(source),
                "--output",
                str(output),
                "--bits",
                "3",
                "--groupsize",
                "24",
                "--sub-bits",
                "5",
                "--row-chunk",
                "4",
                "--quant-backend",
                "cpu",
                "--device",
                "cpu",
                "--tensor-precision-overrides",
                str(overrides),
                "--nvq-codebook-scope",
                "fixed",
            ]
        )
    )

    with open_mmap(output) as store:
        assert store.records["fp8.weight"].dtype == "NINT3"
        assert store.records["fp4.weight"].dtype == "NVQ2"


def test_full_precision_input_rejects_any_mfq_quantized_tensor(tmp_path):
    path = tmp_path / "mixed.mfq"
    tensor = quantize(
        np.arange(4 * 24, dtype=np.float32).reshape(4, 24),
        NintSpec(3, 24, 5),
    )
    save(
        path,
        FileHeader(version=2, model_arch="mixed"),
        {"already_quantized.weight": tensor},
    )
    with pytest.raises(ValueError, match="only full-precision"):
        FullPrecisionMfqCheckpoint(path)
