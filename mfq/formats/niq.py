"""Compatibility shim for :mod:`mfq.formats.nvq`."""

from mfq import _legacy_niq
from mfq.formats import nvq as _implementation

_legacy_niq.export_niq_aliases(globals(), _implementation)
