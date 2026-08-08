"""Streaming access to the official DeepSeek-V4-Flash MX checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mfq.quantize.mxfp import (
    RawSafeTensorFile,
    decode_mxfp4,
    read_dense_rows,
    read_mxfp4_rows,
    read_mxfp8_rows,
)
from mfq.formats.mx import MXFP4_DTYPE, mx_header_bytes


_GLOBAL_NAME_MAP = {
    "embed.weight": "token_embd.weight",
    "head.weight": "output.weight",
    "norm.weight": "output_norm.weight",
    "hc_head_base": "output_hc_base.weight",
    "hc_head_fn": "output_hc_fn.weight",
    "hc_head_scale": "output_hc_scale.weight",
}
_LAYER_NAME_MAP = {
    "hc_attn_base": "hc_attn_base.weight",
    "hc_attn_fn": "hc_attn_fn.weight",
    "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_base": "hc_ffn_base.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight",
    "hc_ffn_scale": "hc_ffn_scale.weight",
    "attn.attn_sink": "attn_sinks.weight",
    "attn.wq_a.weight": "attn_q_a.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight",
    "attn.wq_b.weight": "attn_q_b.weight",
    "attn.wkv.weight": "attn_kv.weight",
    "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_a.weight": "attn_output_a.weight",
    "attn.wo_b.weight": "attn_output_b.weight",
    "attn.compressor.ape": "attn_compressor_ape.weight",
    "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
    "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
    "attn.compressor.norm.weight": "attn_compressor_norm.weight",
    "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "attn.indexer.compressor.ape": "indexer_compressor_ape.weight",
    "attn.indexer.compressor.wgate.weight": "indexer_compressor_gate.weight",
    "attn.indexer.compressor.wkv.weight": "indexer_compressor_kv.weight",
    "attn.indexer.compressor.norm.weight": "indexer_compressor_norm.weight",
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.gate.tid2eid": "ffn_gate_tid2eid.weight",
}


def v4f_source_to_gguf_name(name: str) -> str | None:
    direct = _GLOBAL_NAME_MAP.get(name)
    if direct is not None:
        return direct
    if name.startswith("mtp."):
        return None
    prefix = "layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    layer, separator, suffix = rest.partition(".")
    if not separator or not layer.isdigit():
        return None
    mapped = _LAYER_NAME_MAP.get(suffix)
    return None if mapped is None else f"blk.{layer}.{mapped}"


class V4FCheckpoint:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        index_path = self.root / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = dict(
            json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        )
        self._readers: dict[str, RawSafeTensorFile] = {}

    def shard_for(self, name: str) -> str:
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint tensor is absent: {name}") from exc

    def reader_for(self, name: str) -> RawSafeTensorFile:
        shard = self.shard_for(name)
        reader = self._readers.get(shard)
        if reader is None:
            reader = RawSafeTensorFile(self.root / shard)
            self._readers[shard] = reader
        return reader

    def info(self, name: str):
        return self.reader_for(name).info(name)

    def tensor_source(self, name: str) -> V4FTensorSource:
        return V4FTensorSource(self, name)

    def expert_source(self, layer: int, projection: str) -> V4FExpertSource:
        return V4FExpertSource(self, layer, projection)


class V4FTensorSource:
    """A row-readable logical tensor backed by BF16 or MXFP8 storage."""

    def __init__(self, checkpoint: V4FCheckpoint, name: str) -> None:
        self.checkpoint = checkpoint
        self.name = name
        self.info = checkpoint.info(name)
        self.shape = self.info.shape
        if len(self.shape) != 2:
            raise ValueError(f"row source requires a rank-2 tensor: {name}")

    def reshape(self, *shape: int) -> V4FTensorSource:
        requested = tuple(int(value) for value in shape)
        if requested not in {self.shape, (-1, self.shape[1])}:
            raise ValueError(f"unsupported V4F tensor reshape: {requested}")
        return self

    def read_rows(
        self,
        start: int,
        stop: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        reader = self.checkpoint.reader_for(self.name)
        if self.info.dtype in {"F8_E4M3", "F8_E4M3FN"}:
            scale_name = self.name.removesuffix(".weight") + ".scale"
            if self.checkpoint.shard_for(scale_name) != self.checkpoint.shard_for(
                self.name
            ):
                raise ValueError(f"MXFP8 weight and scale are split: {self.name}")
            return read_mxfp8_rows(
                reader,
                self.name,
                scale_name,
                start,
                stop,
                device=device,
            )
        return read_dense_rows(
            reader,
            self.name,
            start,
            stop,
            device=device,
            dtype=torch.float32,
        )

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("V4F tensor source accepts contiguous slices only")
        start = 0 if key.start is None else int(key.start)
        stop = self.shape[0] if key.stop is None else int(key.stop)
        return self.read_rows(start, stop, device="cpu")

    def read_all_cpu(self, *, preserve_integer: bool = False) -> torch.Tensor:
        reader = self.checkpoint.reader_for(self.name)
        if self.info.dtype in {"F8_E4M3", "F8_E4M3FN"}:
            return self.read_rows(0, self.shape[0], device="cpu")
        return read_dense_rows(
            reader,
            self.name,
            0,
            self.shape[0],
            device="cpu",
            dtype=None if preserve_integer else torch.float32,
        )


class V4FExpertSource:
    """Expose one fused routed-expert projection as ``[256,out,K]``."""

    def __init__(
        self,
        checkpoint: V4FCheckpoint,
        layer: int,
        projection: str,
    ) -> None:
        if projection not in {"gate_up", "down"}:
            raise ValueError(f"unsupported V4F expert projection: {projection}")
        self.checkpoint = checkpoint
        self.layer = int(layer)
        self.projection = projection
        self.n_experts = 256
        self.rows_per_expert = 4096
        self.columns = 4096 if projection == "gate_up" else 2048
        self.shape = (self.n_experts, self.rows_per_expert, self.columns)
        first = self._weight_name(0, "w1" if projection == "gate_up" else "w2")
        info = checkpoint.info(first)
        expected = (
            (2048, 2048) if projection == "gate_up" else (4096, 1024)
        )
        if info.dtype != "I8" or info.shape != expected:
            raise ValueError(
                f"unexpected V4F routed source {first}: {info.dtype} {info.shape}"
            )

    def _weight_name(self, expert: int, part: str) -> str:
        return (
            f"layers.{self.layer}.ffn.experts.{expert}.{part}.weight"
        )

    def _read_part(
        self,
        expert: int,
        part: str,
        start: int,
        stop: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        weight_name = self._weight_name(expert, part)
        scale_name = weight_name.removesuffix(".weight") + ".scale"
        if self.checkpoint.shard_for(weight_name) != self.checkpoint.shard_for(
            scale_name
        ):
            raise ValueError(f"MXFP4 weight and scale are split: {weight_name}")
        return read_mxfp4_rows(
            self.checkpoint.reader_for(weight_name),
            weight_name,
            scale_name,
            start,
            stop,
            device=device,
        )

    def _raw_part(self, expert: int, part: str) -> tuple[np.memmap, np.memmap]:
        """Return the official packed MXFP4 values and E8M0 scales unchanged."""

        weight_name = self._weight_name(expert, part)
        scale_name = weight_name.removesuffix(".weight") + ".scale"
        if self.checkpoint.shard_for(weight_name) != self.checkpoint.shard_for(
            scale_name
        ):
            raise ValueError(f"MXFP4 weight and scale are split: {weight_name}")
        reader = self.checkpoint.reader_for(weight_name)
        weight = reader.info(weight_name)
        scales = reader.info(scale_name)
        expected_values = (
            (2048, 2048) if part in {"w1", "w3"} else (4096, 1024)
        )
        expected_scales = (
            (2048, 128) if part in {"w1", "w3"} else (4096, 64)
        )
        if weight.dtype != "I8" or weight.shape != expected_values:
            raise ValueError(
                f"unexpected MXFP4 value tensor {weight_name}: "
                f"{weight.dtype} {weight.shape}"
            )
        if scales.dtype != "F8_E8M0" or scales.shape != expected_scales:
            raise ValueError(
                f"unexpected MXFP4 scale tensor {scale_name}: "
                f"{scales.dtype} {scales.shape}"
            )
        return reader.raw_tensor(weight_name), reader.raw_tensor(scale_name)

    def write_mxfp4_expert_pool(
        self,
        expert_ids: tuple[int, ...],
        output: str | Path,
    ) -> int:
        """Write selected official experts as one exact native MXFP4 payload."""

        ids = tuple(int(expert) for expert in expert_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("MXFP4 expert pool requires unique expert ids")
        if any(expert < 0 or expert >= self.n_experts for expert in ids):
            raise IndexError("MXFP4 expert pool contains an invalid expert id")
        rows = len(ids) * self.rows_per_expert
        parts = ("w1", "w3") if self.projection == "gate_up" else ("w2",)
        target = Path(output)
        with target.open("wb") as handle:
            handle.write(
                mx_header_bytes(
                    MXFP4_DTYPE,
                    (rows, self.columns),
                    (rows, self.columns // 2),
                    (rows, self.columns // 32),
                )
            )
            for expert in ids:
                for part in parts:
                    values, _ = self._raw_part(expert, part)
                    handle.write(memoryview(values))
            for expert in ids:
                for part in parts:
                    _, scales = self._raw_part(expert, part)
                    handle.write(memoryview(scales))
        return target.stat().st_size

    def read_expert_rows(
        self,
        expert: int,
        start: int,
        stop: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        if expert < 0 or expert >= self.n_experts:
            raise IndexError(f"invalid V4F expert id: {expert}")
        if start < 0 or stop < start or stop > self.rows_per_expert:
            raise IndexError(
                f"invalid V4F expert row slice {start}:{stop}"
            )
        if self.projection == "down":
            return self._read_part(
                expert, "w2", start, stop, device=device
            )
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < stop:
            part = "w1" if cursor < 2048 else "w3"
            local = cursor if cursor < 2048 else cursor - 2048
            take = min(stop - cursor, 2048 - local)
            pieces.append(
                self._read_part(
                    expert,
                    part,
                    local,
                    local + take,
                    device=device,
                )
            )
            cursor += take
        if not pieces:
            return torch.empty(
                (0, self.columns), device=device, dtype=torch.float32
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def read_rows(
        self,
        start: int,
        stop: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        total = self.n_experts * self.rows_per_expert
        if start < 0 or stop < start or stop > total:
            raise IndexError(f"invalid V4F flattened row slice {start}:{stop}")
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < stop:
            expert = cursor // self.rows_per_expert
            local = cursor % self.rows_per_expert
            take = min(stop - cursor, self.rows_per_expert - local)
            pieces.append(
                self.read_expert_rows(
                    expert,
                    local,
                    local + take,
                    device=device,
                )
            )
            cursor += take
        if not pieces:
            return torch.empty(
                (0, self.columns), device=device, dtype=torch.float32
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def reshape(self, *shape: int) -> V4FExpertSource:
        requested = tuple(int(value) for value in shape)
        flattened = (
            self.n_experts * self.rows_per_expert,
            self.columns,
        )
        if requested not in {self.shape, flattened, (-1, self.columns)}:
            raise ValueError(f"unsupported V4F expert reshape: {requested}")
        return self

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("V4F expert source accepts contiguous slices only")
        total = self.n_experts * self.rows_per_expert
        start = 0 if key.start is None else int(key.start)
        stop = total if key.stop is None else int(key.stop)
        return self.read_rows(start, stop, device="cpu")

    @staticmethod
    def _sample_start(
        seed: int,
        layer: int,
        expert: int,
        part: str,
        rows: int,
        total_rows: int,
    ) -> int:
        if total_rows <= 0:
            raise ValueError("source row count must be positive")
        if rows <= 0 or rows > total_rows:
            raise ValueError(
                f"sample row count {rows} must be within [1, {total_rows}]"
            )
        key = f"{seed}:{layer}:{expert}:{part}".encode("ascii")
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        return value % (total_rows - rows + 1)

    def sample_experts(
        self,
        rows: int,
        *,
        seed: int,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Read all expert samples into two host byte arrays and decode once."""

        if rows <= 0:
            raise ValueError("sample row count must be positive")
        if self.projection == "gate_up":
            if rows % 2:
                raise ValueError("gate_up sample rows must split evenly")
            parts = (("w1", rows // 2), ("w3", rows // 2))
        else:
            parts = (("w2", rows),)
        packed_width = self.columns // 2
        scale_width = self.columns // 32
        packed = np.empty(
            (self.n_experts, rows, packed_width), dtype=np.uint8
        )
        scales = np.empty(
            (self.n_experts, rows, scale_width), dtype=np.uint8
        )
        destination = 0
        for part, part_rows in parts:
            for expert in range(self.n_experts):
                weight_name = self._weight_name(expert, part)
                scale_name = weight_name.removesuffix(".weight") + ".scale"
                reader = self.checkpoint.reader_for(weight_name)
                total_rows = reader.info(weight_name).shape[0]
                start = self._sample_start(
                    seed,
                    self.layer,
                    expert,
                    part,
                    part_rows,
                    total_rows,
                )
                packed[expert, destination : destination + part_rows] = (
                    reader.raw_rows(weight_name, start, start + part_rows)
                )
                scales[expert, destination : destination + part_rows] = (
                    reader.raw_rows(scale_name, start, start + part_rows)
                )
            destination += part_rows
        decoded = decode_mxfp4(
            packed.reshape(-1, packed_width),
            scales.reshape(-1, scale_width),
            device=device,
        )
        return decoded.reshape(
            self.n_experts, rows, self.columns
        ).contiguous()


__all__ = [
    "V4FCheckpoint",
    "V4FExpertSource",
    "V4FTensorSource",
    "v4f_source_to_gguf_name",
]
