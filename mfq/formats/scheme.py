"""Precision schemes.

A scheme describes the precision to use for every tensor in the model. It is the output of the calibrator
(``mfq.calibration``) and the input to the quantizer (``mfq.quantize``) and runtime.

Naming convention
--------
MFQ uses a native format string to describe the overall precision profile, for example ``MFQ-W4.51``:

- ``W4.51`` means an average weight precision of 4.51 bits per weight (BPW).

Fractional BPW is achieved through NINT combinations of (bits, groupsize, sub_bits).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mfq.formats.nint import NintSpec


@dataclass
class TensorPlan:
    """Precision plan for one tensor; weight and activation precision may be specified independently."""

    name: str
    weight: NintSpec
    activation: NintSpec | None = None
    hadamard: bool = False


@dataclass
class Scheme:
    """Precision scheme for the complete model."""

    plans: dict[str, TensorPlan] = field(default_factory=dict)

    def label(self, meta: dict[str, tuple[int, int]] | None = None) -> str:
        """Generate an MFQ format string such as ``MFQ-W4.51``.

        ``meta[name] = (numel, neuron_len)`` provides each tensor's element count and neuron-row length
        for calculating its bpw. Return a placeholder string when metadata is omitted.
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
