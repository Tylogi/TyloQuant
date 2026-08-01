"""Compatibility shim for :mod:`mfq.quantize.nvq_jsc`."""

from mfq import _legacy_niq
from mfq.quantize import nvq_jsc as _implementation

_legacy_niq.export_niq_aliases(globals(), _implementation)
