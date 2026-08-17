"""Shared NINT storage profiles for non-training allocation tools."""

from __future__ import annotations

from mfq.formats.nint import NintSpec

NINT_CALIBRATION_PROFILES: dict[str, NintSpec] = {
    "NINT4": NintSpec(4, 24, 6),
    "NINT5": NintSpec(5, 28, 7),
    "NINT6": NintSpec(6, 24, 7),
    "NINT8": NintSpec(8, 48, 7),
}

NINT_EXPERT_PROFILES: dict[str, NintSpec] = {
    "NINT2": NintSpec(2, 16, 5),
    "NINT3": NintSpec(3, 24, 5),
    **NINT_CALIBRATION_PROFILES,
}


def nint_storage_bits(rows: int, columns: int, spec: NintSpec) -> int:
    """Return packed NINT storage in bits for a matrix shape."""

    if rows <= 0 or columns <= 0:
        raise ValueError("NINT storage shape must be positive")
    groups = (columns + spec.groupsize - 1) // spec.groupsize
    return int(
        rows * 32
        + rows * groups * 2 * spec.sub_bits
        + rows * groups * spec.groupsize * spec.bits
    )


__all__ = [
    "NINT_CALIBRATION_PROFILES",
    "NINT_EXPERT_PROFILES",
    "nint_storage_bits",
]
