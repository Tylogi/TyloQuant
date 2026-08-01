"""MFQ 全局配置与常量。

集中放置跨模块共享的枚举与默认值，避免魔法字符串散落。
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    """硬件后端（按计算 API 命名）。"""

    CUDA = "cuda"
    METAL = "metal"
    CPU = "cpu"


# 默认逐层校准 batch 大小
DEFAULT_CALIB_BATCH = 1
