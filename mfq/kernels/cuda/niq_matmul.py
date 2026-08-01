"""Compatibility shim for :mod:`mfq.kernels.cuda.nvq_matmul`."""

from mfq import _legacy_niq
from mfq.kernels.cuda import nvq_matmul as _implementation

_legacy_niq.export_niq_aliases(globals(), _implementation)
