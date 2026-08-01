"""Compatibility entry point for the renamed NVQ codebook trainer."""

from mfq import _legacy_niq
from mfq.tools import train_nvq_codebook as _implementation

_legacy_niq.export_niq_aliases(globals(), _implementation)

if __name__ == "__main__":
    _implementation.main()
