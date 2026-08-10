"""MiniCPM-o 4.5 runtime that preserves the official Transformers graph.

The official remote-code model owns modality preprocessing, SigLIP, the
Resampler, Whisper, Qwen3, TTS, and all cache/mask semantics. Matrix modules
whose checkpoint weights are compact MFQ tensors are replaced with CUDA MFQ
operators. Raw parameters and dense tensors stay attached to the official
modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mfq.formats import io
from mfq.formats.assets import is_asset_record
from mfq.formats.io import MfqTensor
from mfq.formats.mx import MxTensor
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.tpq import TpqInt4Tensor, TpqPqTensor
from mfq.quantize.nint_quant import NintTensor
from mfq.runtime.torch_linear import (
    TorchMxEmbedding,
    TorchMxLinear,
    TorchNint8ZeroEmbedding,
    TorchNint8ZeroLinear,
    TorchNintEmbedding,
    TorchNintLinear,
    TorchNvqEmbedding,
    TorchNvqLinear,
    TorchTpqEmbedding,
    TorchTpqLinear,
    is_nvq_tensor,
    is_quantized_tensor,
)


@dataclass(frozen=True)
class MiniCPMO45LoadReport:
    source_model: str
    mfq_path: str
    official_model_class: str
    quantized_linear_modules: int
    quantized_embedding_modules: int
    dense_state_tensors: int
    checkpoint_tensors: int


class _MfqLinearModule(nn.Module):
    def __init__(
        self,
        tensor: MfqTensor,
        source: nn.Linear,
        *,
        bias: np.ndarray | None,
        device: str | torch.device,
        weight_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        shape = tuple(int(v) for v in tensor.shape)
        expected = (int(source.out_features), int(source.in_features))
        if shape != expected:
            raise ValueError(f"linear weight shape mismatch: MFQ={shape}, module={expected}")
        if isinstance(tensor, NintTensor):
            if int(tensor.axis) != 0:
                raise ValueError("MiniCPM-o linear NINT tensors must use axis=0")
            self._operator = TorchNintLinear(tensor, device)
        elif isinstance(tensor, Nint8ZeroTensor):
            self._operator = TorchNint8ZeroLinear(tensor, device)
        elif is_nvq_tensor(tensor):
            self._operator = TorchNvqLinear(tensor, device)
        elif isinstance(tensor, MxTensor):
            self._operator = TorchMxLinear(tensor, device)
        elif isinstance(tensor, (TpqInt4Tensor, TpqPqTensor)):
            self._operator = TorchTpqLinear(tensor, device)
        else:
            raise TypeError("MiniCPM-o linear weight must be NINT/NVQ/MX/TPQ")
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        self.device = torch.device(device)
        self.register_buffer(
            "_weight_marker",
            torch.empty(0, device=self.device, dtype=weight_dtype),
            persistent=False,
        )
        if bias is None:
            self.register_buffer("bias", None, persistent=False)
        else:
            dense_bias = _dense_to_torch(bias, self.device, dtype=weight_dtype)
            if tuple(dense_bias.shape) != (self.out_features,):
                raise ValueError(
                    f"linear bias shape mismatch: {tuple(dense_bias.shape)} != "
                    f"{(self.out_features,)}"
                )
            self.register_buffer("bias", dense_bias, persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        # Official MiniCPM code reads dtype/device from embedding/linear weights.
        # Packed weights remain owned by the MFQ operator and are never expanded.
        return self._weight_marker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._operator(x)
        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out


class _MfqEmbeddingModule(nn.Module):
    def __init__(
        self,
        tensor: MfqTensor,
        source: nn.Embedding,
        *,
        device: str | torch.device,
        weight_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        shape = tuple(int(v) for v in tensor.shape)
        expected = (int(source.num_embeddings), int(source.embedding_dim))
        if shape != expected:
            raise ValueError(f"embedding weight shape mismatch: MFQ={shape}, module={expected}")
        if source.max_norm is not None:
            raise ValueError("MFQ embedding does not support max_norm during inference")
        if isinstance(tensor, NintTensor):
            if int(tensor.axis) != 0:
                raise ValueError("MiniCPM-o embedding NINT tensors must use axis=0")
            self._operator = TorchNintEmbedding(tensor, device)
        elif isinstance(tensor, Nint8ZeroTensor):
            self._operator = TorchNint8ZeroEmbedding(tensor, device)
        elif is_nvq_tensor(tensor):
            self._operator = TorchNvqEmbedding(tensor, device)
        elif isinstance(tensor, MxTensor):
            self._operator = TorchMxEmbedding(tensor, device)
        elif isinstance(tensor, (TpqInt4Tensor, TpqPqTensor)):
            self._operator = TorchTpqEmbedding(tensor, device)
        else:
            raise TypeError("MiniCPM-o embedding weight must be NINT/NVQ/MX/TPQ")
        self.num_embeddings = int(source.num_embeddings)
        self.embedding_dim = int(source.embedding_dim)
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse
        self.device = torch.device(device)
        self.register_buffer(
            "_weight_marker",
            torch.empty(0, device=self.device, dtype=weight_dtype),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_marker

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self._operator(input)
        if out.dtype != self._weight_marker.dtype:
            out = out.to(dtype=self._weight_marker.dtype)
        return out


class TorchMfqMiniCPMO45:
    """Proxy for the official MiniCPM-o 4.5 model backed by MFQ weights."""

    def __init__(self, model: nn.Module, report: MiniCPMO45LoadReport) -> None:
        self.model = model
        self.load_report = report

    @classmethod
    def from_mfq(
        cls,
        model_dir: str | Path,
        mfq_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        local_files_only: bool = True,
    ) -> TorchMfqMiniCPMO45:
        model, report = _load_minicpmo45(
            Path(model_dir),
            Path(mfq_path),
            device=device,
            local_files_only=local_files_only,
        )
        return cls(model, report)

    def __getattr__(self, name: str) -> Any:
        if name in {"model", "load_report"}:
            raise AttributeError(name)
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_minicpmo45(
    model_dir: str | Path,
    mfq_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> TorchMfqMiniCPMO45:
    return TorchMfqMiniCPMO45.from_mfq(
        model_dir,
        mfq_path,
        device=device,
        local_files_only=local_files_only,
    )


def _load_minicpmo45(
    model_dir: Path,
    mfq_path: Path,
    *,
    device: str | torch.device,
    local_files_only: bool,
) -> tuple[nn.Module, MiniCPMO45LoadReport]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniCPM-o config is missing: {config_path}")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_minicpmo45_config(source_config)

    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "MiniCPM-o 4.5 loading requires its official Transformers dependencies "
            "(transformers==4.51.0, accelerate, and minicpmo-utils)"
        ) from exc

    header, store = io.load_mmap(mfq_path)
    try:
        embedded_config = header.extra.get("hf_config", {})
        if isinstance(embedded_config, dict) and embedded_config:
            _validate_minicpmo45_config(embedded_config)
            _validate_matching_graph_config(source_config, embedded_config)

        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        with init_empty_weights(include_buffers=False):
            model = AutoModel.from_config(
                config,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )

        checkpoint_names = {name for name in store.records if not is_asset_record(name)}
        expected_names = set(model.state_dict())
        missing = sorted(expected_names - checkpoint_names)
        unexpected = sorted(checkpoint_names - expected_names)
        if missing or unexpected:
            raise ValueError(
                "MiniCPM-o checkpoint graph does not match official modules: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )

        module_map = dict(model.named_modules())
        weight_dtype = _torch_dtype_from_config(source_config)
        quantized_names = {
            name for name in checkpoint_names if _record_is_quantized(store.records[name].dtype)
        }
        consumed_quantized: set[str] = set()
        consumed_dense: set[str] = set()
        linear_count = 0
        embedding_count = 0

        for weight_name in sorted(quantized_names):
            if not weight_name.endswith(".weight"):
                raise ValueError(
                    f"quantized raw parameter is unsupported in the official graph: {weight_name}"
                )
            module_name = weight_name[: -len(".weight")]
            source_module = module_map.get(module_name)
            if source_module is None:
                raise ValueError(f"quantized weight has no official module: {weight_name}")
            packed = store[weight_name]
            if not is_quantized_tensor(packed):
                raise TypeError(
                    f"record marked compact did not decode as MFQ weight: {weight_name}"
                )
            if isinstance(source_module, nn.Embedding):
                replacement: nn.Module = _MfqEmbeddingModule(
                    packed,
                    source_module,
                    device=device,
                    weight_dtype=weight_dtype,
                )
                embedding_count += 1
            elif isinstance(source_module, nn.Linear):
                bias_name = f"{module_name}.bias"
                bias = None
                if source_module.bias is not None:
                    if bias_name not in checkpoint_names:
                        raise ValueError(f"linear bias is missing: {bias_name}")
                    bias_value = store[bias_name]
                    if is_quantized_tensor(bias_value):
                        raise TypeError(f"linear bias must be dense: {bias_name}")
                    bias = bias_value
                    consumed_dense.add(bias_name)
                replacement = _MfqLinearModule(
                    packed,
                    source_module,
                    bias=bias,
                    device=device,
                    weight_dtype=weight_dtype,
                )
                linear_count += 1
            else:
                raise TypeError(
                    f"quantized weight is attached to unsupported module "
                    f"{type(source_module).__name__}: {weight_name}"
                )
            _replace_module(model, module_name, replacement)
            consumed_quantized.add(weight_name)

        if consumed_quantized != quantized_names:
            absent = sorted(quantized_names - consumed_quantized)
            raise RuntimeError(f"unconsumed MiniCPM-o quantized tensors: {absent[:8]}")

        current_state_names = set(model.state_dict())
        dense_state: dict[str, torch.Tensor] = {}
        target_device = torch.device(device)
        for name in sorted(current_state_names):
            if name in consumed_dense:
                continue
            if name not in checkpoint_names:
                raise ValueError(f"dense official state is missing from MFQ: {name}")
            value = store[name]
            if is_quantized_tensor(value):
                raise TypeError(f"quantized state was not replaced by a module: {name}")
            dense_state[name] = _dense_to_torch(
                value,
                target_device,
                dtype=weight_dtype,
            )

        load_result = model.load_state_dict(dense_state, strict=False, assign=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError(
                "dense MiniCPM-o state load is incomplete: "
                f"missing={load_result.missing_keys[:8]}, "
                f"unexpected={load_result.unexpected_keys[:8]}"
            )
        del dense_state

        meta_parameters = [
            name for name, parameter in model.named_parameters() if parameter.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                f"official MiniCPM-o parameters remain on meta: {meta_parameters[:8]}"
            )
        meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
        if meta_buffers:
            raise RuntimeError(f"official MiniCPM-o buffers remain on meta: {meta_buffers[:8]}")

        model.to(target_device)
        model.eval()
        report = MiniCPMO45LoadReport(
            source_model=str(model_dir.resolve()),
            mfq_path=str(mfq_path.resolve()),
            official_model_class=type(model).__name__,
            quantized_linear_modules=linear_count,
            quantized_embedding_modules=embedding_count,
            dense_state_tensors=len(checkpoint_names - quantized_names),
            checkpoint_tensors=len(checkpoint_names),
        )
        return model, report
    finally:
        store.close()


def _record_is_quantized(dtype: str) -> bool:
    return dtype.startswith(("NINT", "NVQ", "NPQ"))


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _dense_to_torch(
    value: np.ndarray,
    device: str | torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    is_bfloat16 = io.is_bfloat16_array(value)
    array = np.ascontiguousarray(value)
    if is_bfloat16:
        tensor = torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(array)
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device).contiguous()


def _validate_minicpmo45_config(config: dict[str, Any]) -> None:
    if str(config.get("model_type", "")).lower() != "minicpmo":
        raise ValueError("model_type must be 'minicpmo'")
    if str(config.get("version", "")) != "4.5":
        raise ValueError("only MiniCPM-o version 4.5 is supported")


def _torch_dtype_from_config(config: dict[str, Any]) -> torch.dtype:
    name = str(config.get("torch_dtype", "bfloat16")).lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported MiniCPM-o torch_dtype: {name}")
    return mapping[name]


def _validate_matching_graph_config(
    source: dict[str, Any],
    embedded: dict[str, Any],
) -> None:
    graph_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "init_vision",
        "init_audio",
        "init_tts",
    )
    mismatches = {
        key: (source.get(key), embedded.get(key))
        for key in graph_fields
        if source.get(key) != embedded.get(key)
    }
    if mismatches:
        raise ValueError(f"source and MFQ graph config differ: {mismatches}")


__all__ = [
    "MiniCPMO45LoadReport",
    "TorchMfqMiniCPMO45",
    "load_minicpmo45",
]
