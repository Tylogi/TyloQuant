import re
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "cpp_runtime/backends/metal"
SCRIPT = ROOT / "packaging/compile_metal_libraries.py"


def test_compilation_uses_all_cpus_bounded_by_jobs(monkeypatch):
    workers = runpy.run_path(str(SCRIPT))["worker_count"]
    monkeypatch.setattr("os.cpu_count", lambda: 18)
    assert workers(416) == 18
    assert workers(6) == 6
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert workers(416) == 1
    assert workers(0) == 1


def test_aot_fingerprint_covers_source_options_and_language():
    fingerprint = runpy.run_path(str(SCRIPT))["fingerprint"]
    base = fingerprint(b"source", 2, 4 << 16)
    assert len(base) == 64
    assert base != fingerprint(b"changed", 2, 4 << 16)
    assert base != fingerprint(b"source", 0, 4 << 16)
    assert base != fingerprint(b"source", 2, (3 << 16) + 2)


def test_release_requires_and_bundles_precompiled_libraries():
    script = (ROOT / "packaging/build_release_mac.sh").read_text()
    assert script.index("xcrun --find metal") < script.index("rm -rf")
    assert "-DMFQ_BUILD_PRECOMPILED_METAL=ON" in script
    assert "metal/lib/mfq-metal/manifest.json" in script
    assert '"${mfq_resource_dir}/mfq-metal"' in script
    config = (ROOT / "MFQStudio/src-tauri/tauri.release-macos.conf.json").read_text()
    assert '"../../packaging/Resources/mfq-metal": "mfq-metal"' in config


def test_native_preparation_uses_parallel_cache_for_evaluation_too():
    for path in ["ops/mlx_nint.cpp", "ops/mlx_moe.cpp", "ops/mlx_grouped_linear.cpp",
        "ops/mlx_linear_attention.cpp", "ops/mlx_sparse_attention.cpp", "models/flash_next/qwen4_ops.cpp"]:
        source = (METAL / path).read_text()
        assert "kernel_device(" in source
        for match in re.finditer(r"mlx::core::metal::device\(", source):
            function = source.rfind("\n}", 0, match.start())
            block = source[function:source.find("\n}", match.end())]
            assert path == "ops/mlx_moe.cpp"
            assert any(name in block for name in (
                "bool uses_indirect_expert_pages(",
                "void retain_indirect_resources(",
                "MlxMfeWeight MlxMfeWeight::with_prepared_page_resources(",
            ))
            assert ".get_kernel(" not in block and ".get_library(" not in block
        assert not re.search(r"metal::device\([^;]*\.(?:get_kernel|get_library)\(", source)
    source = (METAL / "runtime/mlx_kernel_cache.cpp").read_text()
    assert source.index("precompiled_library(device_") < source.index("auto config =")
    assert "promise.set_exception(std::current_exception())" in source
    assert "return get_kernel(name, library, name, {});" in source
    assert "return cached(*cache, key," in source
    assert "descriptor->setSpecializedName(" in source
    assert "values->setConstantValue(value, type, index);" in source
    sparse = (METAL / "ops/mlx_sparse_attention.cpp").read_text()
    specialized = sparse.split('const std::string name = "mfq_sdpa_decode_sparse_"', 1)[1].split(";", 2)[1]
    assert "return device.get_kernel(" in specialized
    assert "std::to_string(partitions_)" in specialized
    assert "std::to_string(params_.block_size), constants)" in specialized


def test_compile_command_fails_without_a_source_catalog(tmp_path, monkeypatch):
    script = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), str(tmp_path), str(tmp_path / "out"),
        "--metal", "missing", "--metallib", "missing", "--sdk", "missing"])
    with pytest.raises(SystemExit):
        script["main"]()
