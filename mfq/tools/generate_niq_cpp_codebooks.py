"""Compatibility entry point for the renamed NVQ codebook generator."""

from mfq import _legacy_niq
from mfq.tools import generate_nvq_cpp_codebooks as _implementation

_legacy_niq.export_niq_aliases(globals(), _implementation)

if __name__ == "__main__":
    _implementation.main()
