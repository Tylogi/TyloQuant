"""精度方案（Precision Scheme）。

一个 Scheme 描述「整个模型的每一个 tensor 该用什么精度」。它是校准器
（``mfq.calibration``）的输出、量化器（``mfq.quantize``）与运行时的输入。

命名制式
--------
MFQ 用原生格式串描述整体精度画像，例如 ``MFQ-W4.51``：

- ``W4.51`` 表示权重平均 4.51 bits-per-weight（BPW）

非整数 BPW 由 NINT 的 (bits, groupsize, sub_bits) 组合达成。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mfq.formats.nint import NintSpec


@dataclass
class TensorPlan:
    """单个 tensor 的精度计划。权重与激活可以独立指定。"""

    name: str
    weight: NintSpec
    activation: NintSpec | None = None
    hadamard: bool = False


@dataclass
class Scheme:
    """整个模型的精度方案。"""

    plans: dict[str, TensorPlan] = field(default_factory=dict)

    def label(self, meta: dict[str, tuple[int, int]] | None = None) -> str:
        """生成 MFQ 制式串，如 ``MFQ-W4.51``。

        ``meta[name] = (numel, neuron_len)`` 提供每个 tensor 的元素数与 neuron 行长
        （用于算该 tensor 的 bpw）。未提供时返回占位串。
        """

        if not self.plans:
            return "MFQ-W0.00"
        if meta is None:
            meta = {n: (1, 1) for n in self.plans}
        total = 0
        wsum = 0.0
        for name, plan in self.plans.items():
            numel, nlen = meta.get(name, (0, 1))
            total += numel
            wsum += numel * plan.weight.bpw(nlen)
        bpw = wsum / total if total else 0.0
        return f"MFQ-W{bpw:.2f}"
