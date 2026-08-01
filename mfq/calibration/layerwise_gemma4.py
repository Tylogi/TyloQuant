"""Gemma4 text backend for streamed BF16 reference generation."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from mfq.calibration.layerwise_qwen35 import _OpenSafetensorRows
from mfq.calibration.qwen35 import HfSafetensorIndex
from mfq.calibration.terminal_kl import ChunkedTerminalObjective


class Gemma4LayerwiseBackend:
    """Load one original Gemma4 text decoder layer at a time."""

    teacher_dtype = torch.bfloat16
    quantized_dtype = torch.float16
    reference_logit_range = 64.0

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        attention: str = "sdpa",
    ) -> None:
        from transformers import AutoConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        self.root = Path(model_path).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Gemma4 layerwise replay requested CUDA without CUDA support")
        if attention not in {"eager", "sdpa"}:
            raise ValueError("attention must be eager or sdpa")

        outer_config = AutoConfig.from_pretrained(
            self.root,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.config = getattr(outer_config, "text_config", outer_config)
        if str(getattr(self.config, "model_type", "")) != "gemma4_text":
            raise ValueError(f"expected Gemma4 text config, got {self.config.model_type!r}")
        self.config._attn_implementation = attention
        self.num_layers = int(self.config.num_hidden_layers)
        self.hidden_size = int(self.config.hidden_size)
        self.final_logit_softcapping = float(self.config.final_logit_softcapping)
        self.index = HfSafetensorIndex(self.root)
        self.rotary = Gemma4TextRotaryEmbedding(self.config, device=self.device).to(self.device)
        self._embedding: torch.Tensor | None = None

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            self._embedding = self.index.tensor(
                "model.language_model.embed_tokens.weight",
                device="cpu",
            )
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64)
        hidden = self._embedding.index_select(0, ids.reshape(-1)).reshape(
            *ids.shape,
            self.hidden_size,
        )
        scale = torch.tensor(
            self.hidden_size**0.5,
            dtype=self._embedding.dtype,
            device="cpu",
        )
        return (hidden * scale).to(device=self.device, dtype=self.teacher_dtype)

    def release_initial_state(self) -> None:
        self._embedding = None
        gc.collect()

    def _load_dense_layer(self, layer_index: int) -> nn.Module:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        with torch.device("meta"):
            layer = Gemma4TextDecoderLayer(self.config, layer_index)
        state = dict(self.index.layer_state(layer_index))
        result = layer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Gemma4 layer {layer_index} state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        return layer.to(
            device=self.device,
            dtype=self.teacher_dtype,
        ).eval().requires_grad_(False)

    @contextmanager
    def layer(self, layer_index: int, *, quantized: bool) -> Iterator[nn.Module]:
        if quantized:
            raise NotImplementedError("Gemma4 streamed reference supports BF16 layers only")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Gemma4 layer {layer_index} is out of range")
        layer = self._load_dense_layer(layer_index)
        try:
            yield layer
        finally:
            del layer
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def _load_final_norm(self, dtype: torch.dtype) -> nn.Module:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        with torch.device("meta"):
            norm = Gemma4RMSNorm(
                self.hidden_size,
                eps=float(self.config.rms_norm_eps),
            )
        result = norm.load_state_dict(
            {
                "weight": self.index.tensor(
                    "model.language_model.norm.weight",
                    device="cpu",
                )
            },
            strict=True,
            assign=True,
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Gemma4 final norm state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        return norm.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @contextmanager
    def terminal_objective(self) -> Iterator[ChunkedTerminalObjective]:
        reference_norm = self._load_final_norm(self.teacher_dtype)
        candidate_norm = self._load_final_norm(self.quantized_dtype)
        head_name = (
            "lm_head.weight"
            if "lm_head.weight" in self.index.weight_map
            else "model.language_model.embed_tokens.weight"
        )
        try:
            if self.index.direct_io:
                class _IndexedRows:
                    def __init__(self, index: HfSafetensorIndex, name: str) -> None:
                        self.index = index
                        self.name = name
                        self.shape = index.shape(name)

                    def rows(self, start: int, end: int) -> torch.Tensor:
                        return self.index.tensor(
                            self.name,
                            row_start=start,
                            row_end=end,
                        )

                yield ChunkedTerminalObjective(
                    reference_norm,
                    candidate_norm,
                    _IndexedRows(self.index, head_name),
                )
            else:
                shard = self.root / self.index.weight_map[head_name]
                with safe_open(str(shard), framework="pt", device="cpu") as reader:
                    yield ChunkedTerminalObjective(
                        reference_norm,
                        candidate_norm,
                        _OpenSafetensorRows(reader, head_name),
                    )
        finally:
            del reference_norm, candidate_norm
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    @torch.inference_mode()
    def forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        batch, sequence, _hidden = hidden_states.shape
        position_ids = torch.arange(
            sequence,
            device=self.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(batch, -1)
        layer_type = str(self.config.layer_types[layer_index])
        position_embeddings = self.rotary(hidden_states, position_ids, layer_type)
        mask_fn = (
            create_causal_mask
            if layer_type == "full_attention"
            else create_sliding_window_causal_mask
        )
        attention_mask = mask_fn(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        return layer(
            hidden_states,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )


__all__ = ["Gemma4LayerwiseBackend"]
