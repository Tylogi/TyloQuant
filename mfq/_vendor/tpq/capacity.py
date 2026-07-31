"""不加载 torch 的模型常驻容量计算。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct


def _dense_bf16_all(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "all"}


def _dsv4_bf16_eligible(name: str) -> bool:
    if name in {"head.weight", "embed.weight", "norm.weight"}:
        return True
    if ".ffn.shared_experts." in name or name.endswith("_fn"):
        return True
    if name.endswith(".attn_norm.weight") or name.endswith(".ffn_norm.weight"):
        return True
    if name.endswith(".q_norm.weight") or name.endswith(".kv_norm.weight"):
        return False
    if name.endswith(".norm.weight") or name.endswith(".attn.attn_sink"):
        return False
    return (
        ".attn.indexer." in name
        or ".attn.compressor." in name
        or ".attn." in name
    )


def dsv4_dense_runtime_bytes(
    path: str | Path,
    dense_bf16: str | None,
) -> int | None:
    """返回 ``TPQ_DENSE_BF16=all`` 时 dense 权重的实际 GPU 字节。

    Int4 张量在 safetensors 中的最后一维为逻辑宽度的一半；展开成 BF16 后，
    字节数是打包元素数的四倍。未进入 BF16 常驻组的量化张量保持 q+s，普通
    张量按 store.get_dense 的 FP32 行为计算。
    """
    if not _dense_bf16_all(dense_bf16):
        return None
    source = Path(path)
    with source.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    tensors = {
        name: info
        for name, info in header.items()
        if name != "__metadata__"
    }
    total = 0
    for name, info in tensors.items():
        if name.endswith(".qs"):
            continue
        elements = math.prod(int(value) for value in info["shape"])
        scale = tensors.get(name + ".qs")
        quantized = scale is not None
        if _dsv4_bf16_eligible(name):
            total += elements * (4 if quantized else 2)
        elif quantized:
            total += (
                int(info["data_offsets"][1])
                - int(info["data_offsets"][0])
                + int(scale["data_offsets"][1])
                - int(scale["data_offsets"][0])
            )
        else:
            total += elements * 4
    return total


__all__ = ["dsv4_dense_runtime_bytes"]
