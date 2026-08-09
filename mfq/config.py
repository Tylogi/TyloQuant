"""MFQ global configuration and constants.

Centralize enums and defaults shared across modules to avoid scattered magic strings.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    """Hardware backends named by compute API."""

    CUDA = "cuda"
    METAL = "metal"
    CPU = "cpu"


# Default batch size for per-layer calibration
DEFAULT_CALIB_BATCH = 1
