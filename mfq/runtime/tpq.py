"""TPQ artifact inspection and runtime integration.

The ``cccp-1`` directory spelling remains supported as the legacy source
format used by existing TPQ model archives.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_GIB = 1 << 30


def _cccp_architecture(manifest: dict[str, Any]) -> str:
    config = manifest["config"]
    if (
        str(manifest.get("model_family", "")).lower() == "kimi_k3"
        or ("kda_layers" in config and "routed_hidden" in config)
    ):
        return "kimi_k3"
    return "deepseek_v4" if "hc_mult" in config else "glm"


def _cccp_expert_layers(manifest: dict[str, Any]) -> tuple[int, ...]:
    config = manifest["config"]
    configured = config.get("moe_layers")
    if configured is not None:
        return tuple(sorted(int(layer) for layer in configured))
    layer_count = int(config.get("n_layers", -1))
    first_layer = (
        int(config.get("first_dense_layers", 0))
        if _cccp_architecture(manifest) == "kimi_k3"
        else 0
    )
    return tuple(range(first_layer, layer_count))


@dataclass(frozen=True)
class CCCPArtifact:
    """Validated TPQ model directory (legacy class name)."""

    root: Path
    manifest: dict[str, Any]
    files: tuple[Path, ...]
    disk_bytes: int

    @classmethod
    def open(cls, root: str | os.PathLike[str]) -> CCCPArtifact:
        model_root = Path(root).expanduser().resolve()
        canonical = model_root / "tpq.json"
        legacy = model_root / "cccp.json"
        manifest_path = canonical if canonical.is_file() else legacy
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"TPQ manifest not found: {canonical} or {legacy}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") not in {"tpq-1", "cccp-1"}:
            raise ValueError(
                f"unsupported TPQ format {manifest.get('format')!r}"
            )

        config = manifest.get("config")
        quant = manifest.get("quant")
        if not isinstance(config, dict) or not isinstance(quant, dict):
            raise ValueError("CCCP manifest is missing config or quant metadata")
        expert_files = manifest.get("expert_files")
        routed_layers = (
            (manifest.get("routed_experts") or {}).get("layer_files") or {}
        )
        if expert_files is None and routed_layers:
            expert_files = {
                str(layer): str(item["path"])
                for layer, item in routed_layers.items()
            }
            manifest["expert_files"] = expert_files
        expected_layers = _cccp_expert_layers(manifest)
        actual_layers = (
            tuple(sorted(int(layer) for layer in expert_files))
            if isinstance(expert_files, dict)
            else ()
        )
        if actual_layers != expected_layers:
            raise ValueError(
                "CCCP expert shard layers do not match configured MoE layers: "
                f"{actual_layers} != {expected_layers}"
            )

        dense_files = manifest.get("dense_files")
        if dense_files is None:
            dense_files = [manifest.get("dense_file", "")]
            dense_root = model_root
        else:
            dense_path = str(
                (manifest.get("nonexpert") or {}).get("path", "dense")
            ).strip("/\\")
            normalized = [
                str(value).replace("\\", "/") for value in dense_files
            ]
            prefixed = bool(dense_path) and all(
                value.startswith(dense_path.replace("\\", "/") + "/")
                for value in normalized
            )
            dense_root = model_root if prefixed else model_root / dense_path
        files_list = [
            manifest_path,
            *(dense_root / str(value) for value in dense_files),
            *(
                model_root / str(expert_files[str(layer)])
                for layer in expected_layers
            ),
            *(
                model_root / str(value)
                for value in manifest.get("tokenizer_files", ())
            ),
        ]
        for key in ("dense_audit_file", "dspark_file", "mtp_file"):
            value = manifest.get(key)
            if value:
                files_list.append(model_root / str(value))
        files_list.extend(
            model_root / str(value)
            for value in manifest.get("expert_audit_files", {}).values()
        )
        if any(not str(value) for value in dense_files):
            raise ValueError("CCCP manifest contains an empty file name")

        files = tuple(dict.fromkeys(files_list))
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:4])
            if len(missing) > 4:
                preview += f", ... ({len(missing)} missing)"
            raise FileNotFoundError(f"CCCP artifact is incomplete: {preview}")
        return cls(
            root=model_root,
            manifest=manifest,
            files=files,
            disk_bytes=sum(path.stat().st_size for path in files),
        )

    @property
    def architecture(self) -> str:
        return _cccp_architecture(self.manifest)

    def summary(self) -> dict[str, Any]:
        config = self.manifest["config"]
        quant = self.manifest["quant"]
        tiers = self.manifest.get("tiers_per_layer", {})
        tier_counts: dict[str, int] = {}
        for assignments in tiers.values():
            for tier in str(assignments):
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return {
            "format": self.manifest["format"],
            "architecture": self.architecture,
            "root": str(self.root),
            "disk_bytes": self.disk_bytes,
            "disk_gib": self.disk_bytes / (1 << 30),
            "files": len(self.files),
            "layers": int(config["n_layers"]),
            "experts_per_layer": int(config["n_experts"]),
            "top_k": int(config["top_k"]),
            "dense": quant.get("dense"),
            "int4_group": quant.get("int4_group"),
            "vq": quant.get("vq"),
            "expert_tier_counts": tier_counts,
            "dspark": self.manifest.get("dspark"),
        }

    @property
    def expert_bytes(self) -> int:
        expert_files = self.manifest["expert_files"]
        return sum(
            (self.root / str(filename)).stat().st_size
            for filename in expert_files.values()
        )


def open_cccp_artifact(model: str | os.PathLike[str]):
    """Open either a legacy cccp-1 directory or a native TPQ MFQ file."""

    path = Path(model).expanduser().resolve()
    if path.is_file():
        from mfq.runtime.tpq_mfq import NativeTPQArtifact

        return NativeTPQArtifact.open(path)
    return CCCPArtifact.open(path)


def _read_cgroup_value(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_cgroup_file_cache() -> int:
    try:
        lines = Path("/sys/fs/cgroup/memory.stat").read_text(
            encoding="ascii"
        ).splitlines()
    except OSError:
        return 0
    for line in lines:
        key, separator, value = line.partition(" ")
        if key == "file" and separator:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def configure_cccp_memory(
    artifact: CCCPArtifact,
    *,
    reserve_gib: float = 16.0,
) -> dict[str, int | float | bool | None]:
    """Make TyloQuant memory policy aware of a Linux cgroup-v2 limit."""

    limit = _read_cgroup_value(Path("/sys/fs/cgroup/memory.max"))
    current = _read_cgroup_value(Path("/sys/fs/cgroup/memory.current"))
    if limit is None or current is None:
        return {
            "cgroup_limit_bytes": limit,
            "cgroup_current_bytes": current,
            "headroom_bytes": None,
            "recommended_cache_gib": None,
            "full_resident_disabled": False,
        }

    reclaimable_file = min(current, _read_cgroup_file_cache())
    headroom = max(0, limit - current + reclaimable_file)
    resident_requirement = int(artifact.expert_bytes * 1.05 + 3 * _GIB)
    disabled = False
    if "TPQ_FULL_RESIDENT" not in os.environ and resident_requirement > headroom:
        os.environ["TPQ_FULL_RESIDENT"] = "0"
        disabled = True
    cache_bytes = max(2 * _GIB, headroom - int(reserve_gib * _GIB))
    return {
        "cgroup_limit_bytes": limit,
        "cgroup_current_bytes": current,
        "cgroup_file_cache_bytes": reclaimable_file,
        "headroom_bytes": headroom,
        "recommended_cache_gib": cache_bytes / _GIB,
        "full_resident_disabled": disabled,
    }


def _tpq_candidates() -> tuple[Path, ...]:
    repository_root = Path(__file__).resolve().parents[2]
    package_root = Path(__file__).resolve().parents[1]
    candidates = [
        package_root / "_vendor" / "tpq",
        repository_root / "references" / "tyloquant-pq",
        Path.cwd() / "references" / "tyloquant-pq",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _clear_tpq_modules() -> None:
    for name in tuple(sys.modules):
        if name == "tpq" or name.startswith("tpq."):
            sys.modules.pop(name, None)


def _load_tpq_directory(candidate: Path) -> ModuleType:
    candidate = candidate.expanduser().resolve()
    package_init = candidate / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            f"TyloQuant PQ package entry point not found: {package_init}"
        )
    spec = importlib.util.spec_from_file_location(
        "tpq",
        package_init,
        submodule_search_locations=[str(candidate)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create a TyloQuant PQ module spec for {candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tpq"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _clear_tpq_modules()
        raise
    return module


def load_tpq_package(
    root: str | os.PathLike[str] | None = None,
) -> ModuleType:
    """Load an explicit, installed, or MFQ-vendored TyloQuant runtime."""

    selected = root if root is not None else os.environ.get("MFQ_TPQ_ROOT")
    if selected:
        selected_path = Path(selected).expanduser().resolve()
        loaded = sys.modules.get("tpq")
        loaded_file = getattr(loaded, "__file__", None)
        if (
            loaded_file is not None
            and Path(loaded_file).resolve().parent == selected_path
        ):
            return loaded
        _clear_tpq_modules()
        return _load_tpq_directory(selected_path)

    loaded = sys.modules.get("tpq")
    if loaded is not None:
        return loaded

    installed = importlib.util.find_spec("tpq")
    if installed is not None:
        return importlib.import_module("tpq")

    for candidate in _tpq_candidates():
        if (candidate / "__init__.py").is_file():
            return _load_tpq_directory(candidate)
    raise ModuleNotFoundError(
        "TyloQuant PQ was not found in the MFQ package. Install tpq or set "
        "MFQ_TPQ_ROOT to its package directory."
    )


def load_cccp_model(
    model: str | os.PathLike[str],
    *,
    device: str = "cuda",
    cache_gb: float = 16.0,
    vram_gb: float = 12.0,
    max_ctx: int = 4096,
    tp_size: int = 1,
    tpq_root: str | os.PathLike[str] | None = None,
):
    """Load a CCCP model while retaining compressed expert weights."""

    artifact = open_cccp_artifact(model)
    configure_cccp_memory(artifact)
    if str(device).lower() in {"metal", "mlx", "mps"}:
        if artifact.architecture not in {"kimi_k3", "deepseek_v4"}:
            raise ValueError(
                "the MLX CCCP entry point supports Kimi-K3 and DeepSeek-V4"
            )
        if not hasattr(artifact, "path"):
            raise ValueError(
                "Metal CCCP deployment requires a native CCCP MFQ file; "
                "import the TPQ2 directory with MFQ first"
            )
        if artifact.architecture == "kimi_k3":
            from mfq.runtime.mlx_kimi_k3 import MlxKimiK3

            return MlxKimiK3.from_mfq(artifact.path)
        from mfq.runtime.mlx_deepseek_v4 import MlxDeepseekV4

        return MlxDeepseekV4.from_mfq(
            artifact.path,
            max_context=max_ctx,
            expert_cache_gb=cache_gb,
        )
    tpq = load_tpq_package(tpq_root)
    from mfq.runtime.tpq_residency_patch import apply_tpq_residency_patch

    apply_tpq_residency_patch()
    if hasattr(artifact, "path"):
        from mfq.runtime.tpq_mfq import install_mfq_tpq_store

        install_mfq_tpq_store(tpq)
    artifact_path = (
        artifact.path if hasattr(artifact, "path") else artifact.root
    )
    if artifact.architecture == "deepseek_v4":
        from tpq.dsv4model import DSV4TPQModel

        runtime = DSV4TPQModel(
            str(artifact_path),
            cache_gb=cache_gb,
            max_ctx=max_ctx,
            device=device,
            vram_cache_gb=vram_gb,
            tp_size=tp_size,
        )
    elif artifact.architecture == "kimi_k3":
        from tpq.kimi_model import KimiK3TPQModel

        runtime = KimiK3TPQModel(
            str(artifact_path),
            cache_gb=cache_gb,
            max_ctx=max_ctx,
            device=device,
            vram_cache_gb=vram_gb,
            tp_size=tp_size,
        )
    else:
        raise ValueError(
            "load_cccp_model supports DeepSeek-V4 and Kimi-K3 CCCP artifacts"
        )
    runtime.preload()
    return runtime


@contextmanager
def _native_tokenizer_host(artifact, tokenizer_root: str | os.PathLike[str]):
    root = Path(tokenizer_root).expanduser().resolve()
    tokenizer = root / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer}")
    requested = {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
        *(
            Path(value).name
            for value in artifact.manifest.get("tokenizer_files", ())
        ),
    }
    with tempfile.TemporaryDirectory(prefix="mfq-cccp-chat-") as temporary:
        host = Path(temporary)
        legacy_manifest = dict(artifact.manifest)
        legacy_manifest["format"] = "cccp-1"
        (host / "cccp.json").write_text(
            json.dumps(legacy_manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        for name in sorted(requested):
            source = root / name
            if source.is_file():
                shutil.copy2(source, host / name)
        yield host


def _pop_argument(arguments: list[str], option: str) -> str | None:
    try:
        index = arguments.index(option)
    except ValueError:
        return None
    if index + 1 >= len(arguments):
        raise ValueError(f"{option} requires a value")
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def run_cccp_chat(
    argv: Sequence[str],
    *,
    tpq_root: str | os.PathLike[str] | None = None,
) -> None:
    """Run TyloQuant's production CCCP chat/generation entry point."""

    arguments = list(argv)
    tokenizer_root = _pop_argument(arguments, "--tokenizer-root")
    try:
        model_index = arguments.index("--model") + 1
        artifact = open_cccp_artifact(arguments[model_index])
    except (ValueError, IndexError):
        artifact = None
    if artifact is not None:
        configure_cccp_memory(artifact)
    tpq = load_tpq_package(tpq_root)
    from mfq.runtime.tpq_residency_patch import apply_tpq_residency_patch

    apply_tpq_residency_patch()
    from tpq.chat import main as chat_main

    if artifact is None or not hasattr(artifact, "path"):
        chat_main(arguments)
        return
    if tokenizer_root is None:
        raise ValueError("native CCCP MFQ chat requires --tokenizer-root")
    from mfq.runtime.tpq_mfq import install_mfq_tpq_store

    install_mfq_tpq_store(tpq)
    engine_module = importlib.import_module("tpq.engine")
    original_make_model = engine_module._make_model

    def make_native_model(
        _model_dir: str,
        cache_gb: float,
        max_ctx: int,
        device: str,
        vram_cache_gb: float,
        tp_size: int = 1,
        extreme_fixed_gpu_bytes: int = 0,
    ):
        if artifact.architecture == "kimi_k3":
            from tpq.kimi_model import KimiK3TPQModel

            model_type = KimiK3TPQModel
            architecture = "kimi_k3"
        else:
            from tpq.dsv4model import DSV4TPQModel

            model_type = DSV4TPQModel
            architecture = "dsv4"
        return (
            model_type(
                str(artifact.path),
                cache_gb=cache_gb,
                max_ctx=max_ctx,
                device=device,
                vram_cache_gb=vram_cache_gb,
                tp_size=tp_size,
                extreme_fixed_gpu_bytes=extreme_fixed_gpu_bytes,
            ),
            architecture,
        )

    with _native_tokenizer_host(artifact, tokenizer_root) as host:
        arguments[model_index] = str(host)
        engine_module._make_model = make_native_model
        try:
            chat_main(arguments)
        finally:
            engine_module._make_model = original_make_model


# Canonical API names.  Historical names remain callable aliases.
TPQArtifact = CCCPArtifact
open_tpq_artifact = open_cccp_artifact
configure_tpq_memory = configure_cccp_memory
load_tpq_model = load_cccp_model
run_tpq_chat = run_cccp_chat
