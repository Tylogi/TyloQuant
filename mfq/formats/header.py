"""MFQ 文件头与张量级元数据。

一个 MFQ 文件由一个全局头 + 若干张量记录组成。头部记录魔数、版本、模型
结构摘要；每个张量记录记录其精度规格（NINT 变体、groupsize 等）以及压缩后
的权重 blob 位置。

具体二进制布局见 :mod:`mfq.formats.io`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MFQ_MAGIC = b"MFQ1"


@dataclass
class FileHeader:
    """MFQ 文件全局头。"""

    magic: bytes = MFQ_MAGIC
    version: int = 1
    model_arch: str = ""
    num_tensors: int = 0
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class TensorRecord:
    """单个张量在文件中的定位与精度元数据。"""

    name: str
    dtype: str          # "NINT4" / "NINT5" ...
    shape: tuple[int, ...]
    offset: int         # 权重 blob 在文件中的字节偏移
    nbytes: int         # 权重 blob 字节数
    groupsize: int = 0
    hadamard: bool = False
