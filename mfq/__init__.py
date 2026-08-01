"""MFQ — Mixed Format Quantization.

面向 Blackwell 与 Apple Silicon 的 tensorwise 混合精度量化工具链。

子包总览
--------
formats
    MFQ 原生格式定义（NINT neuron-anchored INT 编解码、精度方案 scheme、文件头与序列化）。
quantize
    量化执行逻辑（NINT 张量量化、灵敏度分析与精度分配、Hadamard 旋转）。
calibration
    逐层校准（收集全精度与量化路径的 hidden、按层分配精度）。
kernels
    硬件后端 kernel（CUDA / Metal；torch GPU 参考实现见 runtime）。
runtime
    推理引擎。
utils
    通用工具（日志、张量辅助）。
"""

from mfq._version import __version__

__all__ = ["__version__"]
