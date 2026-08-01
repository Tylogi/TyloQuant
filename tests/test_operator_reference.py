from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "developer-architecture-and-operator-reference.md"


def _source_bindings(path: Path) -> set[str]:
    return set(
        re.findall(
            r'm\.def\(\s*"([^"]+)"',
            path.read_text(encoding="utf-8"),
        )
    )


def _documented_bindings(begin: str, end: str) -> set[str]:
    text = REFERENCE.read_text(encoding="utf-8")
    section = text.split(begin, 1)[1].split(end, 1)[0]
    return set(re.findall(r"^- `([^`]+)`", section, flags=re.MULTILINE))


def test_native_cuda_bindings_are_documented() -> None:
    source = _source_bindings(ROOT / "mfq" / "kernels" / "cuda" / "mfq_cuda.cpp")
    documented = _documented_bindings(
        "<!-- MFQ_NATIVE_BINDINGS_BEGIN -->",
        "<!-- MFQ_NATIVE_BINDINGS_END -->",
    )
    assert documented == source


def test_bundled_tpq_cuda_bindings_are_documented() -> None:
    source = _source_bindings(ROOT / "mfq" / "_vendor" / "tpq" / "csrc" / "vq_gemv.cu")
    documented = _documented_bindings(
        "<!-- MFQ_TPQ_BINDINGS_BEGIN -->",
        "<!-- MFQ_TPQ_BINDINGS_END -->",
    )
    assert documented == source


def test_offline_quantization_cuda_bindings_are_documented() -> None:
    source = _source_bindings(
        ROOT / "mfq" / "quantize" / "cuda" / "nvq_quant_cuda.cpp"
    )
    documented = _documented_bindings(
        "<!-- MFQ_QUANT_BINDINGS_BEGIN -->",
        "<!-- MFQ_QUANT_BINDINGS_END -->",
    )
    assert documented == source


def test_native_runtime_only_cuda_entrypoints_are_documented() -> None:
    runtime = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")
    declarations = runtime.split("struct Record", 1)[0]
    entrypoints = set(
        re.findall(
            r"^(?:torch::Tensor|std::vector<torch::Tensor>|void)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            declarations,
            flags=re.MULTILINE,
        )
    )
    python_bindings = _source_bindings(
        ROOT / "mfq" / "kernels" / "cuda" / "mfq_cuda.cpp"
    )
    runtime_only = entrypoints - python_bindings - {"mfq_set_env"}
    documented = _documented_bindings(
        "<!-- MFQ_RUNTIME_ONLY_BINDINGS_BEGIN -->",
        "<!-- MFQ_RUNTIME_ONLY_BINDINGS_END -->",
    )
    assert documented == runtime_only


def test_bundled_tpq_environment_switches_are_documented() -> None:
    source_names: set[str] = set()
    for path in (ROOT / "mfq" / "_vendor" / "tpq").rglob("*"):
        if path.suffix not in {".py", ".cu", ".cpp", ".h"}:
            continue
        source_names.update(
            re.findall(r"\bTPQ_[A-Z0-9_]+\b", path.read_text(encoding="utf-8"))
        )
    reference = REFERENCE.read_text(encoding="utf-8")
    documented = set(re.findall(r"`(TPQ_[A-Z0-9_]+)`", reference))
    assert documented == source_names


def test_cuda_source_kernel_counts_are_documented() -> None:
    source_paths = (
        list((ROOT / "mfq" / "kernels" / "cuda").glob("*.cu"))
        + list((ROOT / "mfq" / "quantize" / "cuda").glob("*.cu"))
        + list((ROOT / "mfq" / "_vendor" / "tpq" / "csrc").glob("*.cu"))
    )
    kernel_pattern = re.compile(
        r"__global__\s+"
        r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
        r"(?:[\w:<>,*&\s]+?)\s+([A-Za-z_]\w*)\s*\("
    )
    source_counts = {
        path.relative_to(ROOT).as_posix(): len(
            kernel_pattern.findall(path.read_text(encoding="utf-8"))
        )
        for path in source_paths
    }

    text = REFERENCE.read_text(encoding="utf-8")
    section = text.split("<!-- MFQ_CUDA_SOURCE_TABLE_BEGIN -->", 1)[1].split(
        "<!-- MFQ_CUDA_SOURCE_TABLE_END -->", 1
    )[0]
    documented_counts = {
        path: int(count)
        for path, count in re.findall(
            r"^\| `([^`]+\.cu)` \| (\d+) \|",
            section,
            flags=re.MULTILINE,
        )
    }
    assert documented_counts == source_counts
