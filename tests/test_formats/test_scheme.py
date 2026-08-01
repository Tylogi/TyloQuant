"""scheme 精度方案与 label 测试。"""

from __future__ import annotations

from mfq.formats.nint import NintSpec
from mfq.formats.scheme import Scheme, TensorPlan


def test_empty_scheme_label():
    assert Scheme().label() == "MFQ-W0.00"


def test_label_weighted_bpw():
    # 两个 tensor：A 1e6 元素 @ (4,24,6)->4.506bpw；B 1e6 @ (4,32,8)->4.506bpw
    sch = Scheme(plans={
        "A": TensorPlan(name="A", weight=NintSpec(4, 24, 6)),
        "B": TensorPlan(name="B", weight=NintSpec(4, 32, 8)),
    })
    meta = {"A": (1_000_000, 5120), "B": (1_000_000, 5120)}
    lbl = sch.label(meta)
    assert lbl.startswith("MFQ-W")
    bpw = float(lbl.split("-W")[1])
    expected = (NintSpec(4, 24, 6).bpw(5120) + NintSpec(4, 32, 8).bpw(5120)) / 2
    assert abs(bpw - expected) < 0.01


def test_tensorplan_defaults():
    p = TensorPlan(name="x", weight=NintSpec())
    assert p.activation is None and p.hadamard is False
