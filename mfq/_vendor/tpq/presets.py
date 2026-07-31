"""模型识别与发布预设加载。

发布入口只依赖模型目录中的 ``cccp.json``，不依赖模型文件名：

* 含 ``hc_mult`` 或 ``compress_ratios`` 的模型识别为 DeepSeek-V4；
* 其他当前 CCCP MoE 模型识别为 GLM。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).with_name("configs")


@dataclass(frozen=True)
class ResolvedPreset:
    model_dir: Path
    manifest: dict[str, Any]
    architecture: str
    display_name: str
    profile: str
    config_profile: str
    tp: int
    ep_layout: str | None
    defaults: dict[str, Any]
    environment: dict[str, str]
    supports_parallel: bool


def load_manifest(model_dir: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    root = Path(model_dir).expanduser().resolve()
    manifest_path = root / "cccp.json"
    if not root.is_dir():
        raise ValueError(f"模型目录不存在：{root}")
    if not manifest_path.is_file():
        raise ValueError(f"模型目录缺少 cccp.json：{root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "cccp-1":
        raise ValueError(
            f"不支持的模型格式：{manifest.get('format')!r}，需要 'cccp-1'"
        )
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("cccp.json 缺少 config 对象")
    return root, manifest


def detect_architecture(manifest: dict[str, Any]) -> str:
    config = manifest["config"]
    if (
        str(manifest.get("model_family", "")).lower() == "kimi_k3"
        or ("kda_layers" in config and "routed_hidden" in config)
    ):
        return "kimi_k3"
    if "hc_mult" in config or "compress_ratios" in config:
        return "dsv4"
    return "glm"


def load_arch_config(architecture: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{architecture}.json"
    if not path.is_file():
        raise ValueError(f"没有架构配置：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != 1 or data.get("architecture") != architecture:
        raise ValueError(f"架构配置格式错误：{path}")
    return data


def choose_ep_layout(manifest: dict[str, Any], tp: int) -> str:
    """选择多卡专家布局；不能整除时自动改用 expert-ID 分片。"""
    if tp < 2:
        raise ValueError("并行布局要求 tp >= 2")
    if detect_architecture(manifest) == "kimi_k3":
        # This is an operator layout selected by configuration, not a
        # separate Kimi execution system.
        return "tensor"
    config = manifest["config"]
    intermediate = int(config["moe_inter"])
    dims = {
        int(value[0])
        for value in manifest.get("quant", {}).get("vq", {}).values()
    }
    tensor_ok = intermediate % tp == 0
    if tensor_ok:
        local = intermediate // tp
        tensor_ok = all(local % dim == 0 for dim in dims)
    return "tensor" if tensor_ok else "expert"


def resolve_preset(
    model_dir: str | os.PathLike[str],
    *,
    profile: str = "auto",
    tp: int | None = None,
) -> ResolvedPreset:
    root, manifest = load_manifest(model_dir)
    architecture = detect_architecture(manifest)
    config = load_arch_config(architecture)
    supports_parallel = bool(config.get("supports_parallel", False))

    if profile not in {"auto", "ram", "parallel"}:
        raise ValueError(f"未知 profile：{profile}")
    if profile == "auto":
        profile = (
            "parallel"
            if supports_parallel and tp is not None and tp > 1
            else "ram"
        )
    if profile == "parallel" and not supports_parallel:
        raise ValueError(
            f"{config['display_name']} 当前没有多卡执行路径；请使用 --profile ram --tp 1"
        )
    profiles = config.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"{architecture} 没有 profile={profile!r}")

    selected = profiles[profile]
    resolved_tp = int(selected.get("tp", 1) if tp is None else tp)
    if resolved_tp <= 0:
        raise ValueError("tp 必须为正整数")
    if profile == "ram" and resolved_tp != 1:
        raise ValueError("RAM profile 固定使用 tp=1；多卡请选择 --profile parallel")
    if profile == "parallel" and resolved_tp < 2:
        raise ValueError("parallel profile 要求 tp >= 2")

    config_profile = profile
    if profile == "parallel":
        tp_profile = f"parallel_tp{resolved_tp}"
        if tp_profile in profiles:
            selected = profiles[tp_profile]
            configured_tp = int(selected.get("tp", resolved_tp))
            if configured_tp != resolved_tp:
                raise ValueError(
                    f"{architecture} profile={tp_profile!r} has tp="
                    f"{configured_tp}, requested tp={resolved_tp}"
                )
            config_profile = tp_profile

    environment = {
        str(key): str(value)
        for key, value in config.get("environment", {}).items()
    }
    environment.update(
        {
            str(key): str(value)
            for key, value in selected.get("environment", {}).items()
        }
    )
    ep_layout = (
        choose_ep_layout(manifest, resolved_tp)
        if profile == "parallel"
        else None
    )

    return ResolvedPreset(
        model_dir=root,
        manifest=manifest,
        architecture=architecture,
        display_name=str(config["display_name"]),
        profile=profile,
        config_profile=config_profile,
        tp=resolved_tp,
        ep_layout=ep_layout,
        defaults=dict(config.get("defaults", {})),
        environment=environment,
        supports_parallel=supports_parallel,
    )


def apply_preset_environment(
    preset: ResolvedPreset,
) -> dict[str, str]:
    """Apply a resolved profile without overriding explicit user choices."""
    effective: dict[str, str] = {}
    for key, value in preset.environment.items():
        effective[key] = os.environ.setdefault(key, value)

    if preset.ep_layout is not None:
        configured_layout = os.environ.get("TPQ_EP_LAYOUT")
        if (
            configured_layout == "tensor" and
            preset.ep_layout != "tensor"
        ):
            raise ValueError(
                f"tp={preset.tp} 不能整除该模型专家中间维；"
                "请取消 TPQ_EP_LAYOUT=tensor 或改用 expert"
            )
        effective["TPQ_EP_LAYOUT"] = os.environ.setdefault(
            "TPQ_EP_LAYOUT",
            preset.ep_layout,
        )
    return effective


__all__ = [
    "ResolvedPreset",
    "apply_preset_environment",
    "choose_ep_layout",
    "detect_architecture",
    "load_arch_config",
    "load_manifest",
    "resolve_preset",
]
