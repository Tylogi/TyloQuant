from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "docs" / "llamacpp-developer-architecture-and-operator-reference.md"
)


def _reference_root() -> Path:
    configured = os.environ.get("LLAMA_CPP_REFERENCE_DIR")
    candidates = [Path(configured)] if configured else []
    for path in candidates:
        if path.is_dir() and (path / "ggml" / "include" / "ggml.h").is_file():
            return path
    pytest.skip("llama.cpp reference checkout is not available")


def _section(begin: str, end: str) -> str:
    text = REFERENCE.read_text(encoding="utf-8")
    return text.split(begin, 1)[1].split(end, 1)[0]


def _enum_members(source: str, enum_name: str, prefix: str) -> list[str]:
    start = source.index(f"enum {enum_name} {{")
    end = source.index("\n    };", start)
    block = re.sub(r"//.*", "", source[start:end])
    members: list[str] = []
    for member in re.findall(rf"\b({re.escape(prefix)}[A-Z0-9_]+)\b", block):
        if member.endswith("_COUNT") or member in members:
            continue
        members.append(member)
    return members


def _documented_ops() -> dict[str, bool]:
    section = _section(
        "<!-- LLAMA_GGML_OPS_BEGIN -->",
        "<!-- LLAMA_GGML_OPS_END -->",
    )
    return {
        name: cuda == "是"
        for name, cuda in re.findall(
            r"^\| `(GGML_OP_[A-Z0-9_]+)` \| [^|]+ \| 是 \| (是|否) \|$",
            section,
            flags=re.MULTILINE,
        )
    }


def _documented_types() -> dict[str, tuple[bool, bool]]:
    section = _section(
        "<!-- LLAMA_GGML_TYPES_BEGIN -->",
        "<!-- LLAMA_GGML_TYPES_END -->",
    )
    result: dict[str, tuple[bool, bool]] = {}
    for line in section.splitlines():
        match = re.match(
            r"^\| `(GGML_TYPE_[A-Z0-9_]+)` \| "
            r"\d+ \| \d+ \| [0-9.]+ \| [^|]+ \| (是|否) \| (是|否) \|$",
            line,
        )
        if match:
            result[match.group(1)] = (
                match.group(2) == "是",
                match.group(3) == "是",
            )
    return result


def _function_block(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_document_has_complete_internal_inventories() -> None:
    metadata = _section(
        "<!-- LLAMA_REFERENCE_METADATA_BEGIN -->",
        "<!-- LLAMA_REFERENCE_METADATA_END -->",
    )
    assert "`LLAMA_GGML_OP_COUNT=97`" in metadata
    assert "`LLAMA_GGML_TYPE_COUNT=34`" in metadata
    assert len(_documented_ops()) == 97
    assert len(_documented_types()) == 34


def test_reference_revision_and_model_inventory_are_current() -> None:
    ref = _reference_root()
    metadata = _section(
        "<!-- LLAMA_REFERENCE_METADATA_BEGIN -->",
        "<!-- LLAMA_REFERENCE_METADATA_END -->",
    )
    values = dict(re.findall(r"`([A-Z_]+)=([^`]+)`", metadata))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ref,
        text=True,
    ).strip()
    assert values["LLAMA_REFERENCE_HEAD"] == head

    model_source = (ref / "src" / "llama-model.cpp").read_text(encoding="utf-8")
    mapping = _function_block(
        model_source,
        "static llama_model * llama_model_mapping",
        "\nstatic bool qwen35_mixed_kv_enabled",
    )
    architectures = set(re.findall(r"case (LLM_ARCH_[A-Z0-9_]+):", mapping))
    assert int(values["LLAMA_MODEL_ARCH_COUNT"]) == len(architectures)
    assert int(values["LLAMA_MODEL_SOURCE_COUNT"]) == len(
        list((ref / "src" / "models").glob("*.cpp"))
    )


def test_ggml_ops_and_cuda_dispatch_are_documented() -> None:
    ref = _reference_root()
    header = (ref / "ggml" / "include" / "ggml.h").read_text(encoding="utf-8")
    cuda = (ref / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu").read_text(
        encoding="utf-8"
    )
    ops = _enum_members(header, "ggml_op", "GGML_OP_")
    dispatch = _function_block(
        cuda,
        "static bool ggml_cuda_compute_forward",
        "\n////////////////////////////////////////////////////////////////////////////////\n\n// backend",
    )
    cuda_ops = set(re.findall(r"case (GGML_OP_[A-Z0-9_]+):", dispatch))
    expected = {op: op in cuda_ops for op in ops}
    assert _documented_ops() == expected


def test_ggml_types_and_quant_matmul_support_are_documented() -> None:
    ref = _reference_root()
    header = (ref / "ggml" / "include" / "ggml.h").read_text(encoding="utf-8")
    types = _enum_members(header, "ggml_type", "GGML_TYPE_")

    mmvq_source = (ref / "ggml" / "src" / "ggml-cuda" / "mmvq.cu").read_text(
        encoding="utf-8"
    )
    mmq_source = (ref / "ggml" / "src" / "ggml-cuda" / "mmq.cu").read_text(
        encoding="utf-8"
    )
    mmvq = _function_block(
        mmvq_source,
        "static void mul_mat_vec_q_switch_type",
        "\nvoid ggml_cuda_mul_mat_vec_q",
    )
    mmq = _function_block(
        mmq_source,
        "static void ggml_cuda_mul_mat_q_switch_type",
        "\nvoid ggml_cuda_mul_mat_q",
    )
    mmvq_types = set(re.findall(r"case (GGML_TYPE_[A-Z0-9_]+):", mmvq))
    mmq_types = set(re.findall(r"case (GGML_TYPE_[A-Z0-9_]+):", mmq))
    expected = {
        type_name: (type_name in mmvq_types, type_name in mmq_types)
        for type_name in types
    }
    assert _documented_types() == expected


def test_default_cuda_translation_unit_count_is_current() -> None:
    ref = _reference_root()
    cuda = ref / "ggml" / "src" / "ggml-cuda"
    translation_units = set(cuda.glob("*.cu"))
    translation_units.update((cuda / "template-instances").glob("fattn-tile*.cu"))
    translation_units.update((cuda / "template-instances").glob("fattn-mma*.cu"))
    translation_units.update((cuda / "template-instances").glob("mmq*.cu"))
    translation_units.update((cuda / "template-instances").glob("mmf*.cu"))
    for name in (
        "fattn-vec-instance-f16-f16.cu",
        "fattn-vec-instance-q4_0-q4_0.cu",
        "fattn-vec-instance-q8_0-q8_0.cu",
        "fattn-vec-instance-bf16-bf16.cu",
    ):
        translation_units.add(cuda / "template-instances" / name)

    metadata = _section(
        "<!-- LLAMA_REFERENCE_METADATA_BEGIN -->",
        "<!-- LLAMA_REFERENCE_METADATA_END -->",
    )
    expected = int(
        re.search(r"`LLAMA_CUDA_DEFAULT_TU_COUNT=(\d+)`", metadata).group(1)
    )
    assert len(translation_units) == expected


def test_local_kld_protocol_extensions_are_documented() -> None:
    ref = _reference_root()
    source = (
        ref / "tools" / "perplexity" / "perplexity.cpp"
    ).read_text(encoding="utf-8")
    document = REFERENCE.read_text(encoding="utf-8")
    for marker in ("_logit3_", "sum_reverse_kld", "MFQ_PPL_CHUNK_OFFSET"):
        assert marker in source
        assert marker in document
