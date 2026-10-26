"""Additional disk allocations, grouped by the destination filesystem."""

from __future__ import annotations

import errno
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskSpaceRequirement:
    """Bytes still to allocate; existing files are already charged to free space."""

    path: Path
    nbytes: int
    purpose: str


def filesystem(path: Path) -> tuple[int, Path]:
    """Resolve junctions/symlinks and use the nearest existing directory's device."""

    existing = Path(path).resolve()
    while not existing.exists():
        existing = existing.parent
    if not existing.is_dir():
        existing = existing.parent
    return existing.stat().st_dev, existing


def disk_space_headroom(nbytes: int) -> int:
    """Reserve 5% plus 64 MiB for an estimated filesystem allocation."""
    return (nbytes + 19) // 20 + (64 << 20)


def check_disk_space(
    requirements: Sequence[DiskSpaceRequirement],
    *,
    stage: str,
    margin: bool = True,
) -> None:
    """Fail before work starts when any filesystem cannot hold its allocations.

    Estimates reserve 5% plus 64 MiB per filesystem for variable codec tables,
    container metadata, allocation granularity and small calibration sidecars.
    This is a snapshot, not a reservation against other writers or quotas.
    """

    groups: dict[int, tuple[Path, list[DiskSpaceRequirement]]] = {}
    for requirement in requirements:
        if requirement.nbytes < 0:
            raise ValueError("disk space requirements must be non-negative")
        if requirement.nbytes == 0:
            continue
        device, directory = filesystem(requirement.path)
        groups.setdefault(device, (directory, []))[1].append(requirement)
    reports = []
    insufficient = False
    for directory, items in groups.values():
        allocation = sum(item.nbytes for item in items)
        reserve = disk_space_headroom(allocation) if margin else 0
        required = allocation + reserve
        free = shutil.disk_usage(directory).free
        insufficient |= free < required
        detail = "; ".join(
            f"{item.purpose}: {item.nbytes / 1e9:.3f} GB ({item.nbytes} bytes) at {item.path}"
            for item in items
        )
        reports.append(
            f"  filesystem at {directory}: need {required / 1e9:.3f} GB ({required} bytes), "
            f"free {free / 1e9:.3f} GB ({free} bytes); {detail}; "
            f"headroom {reserve / 1e9:.3f} GB"
        )
    if insufficient:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient disk space before {stage}:\n" + "\n".join(reports)
            + "\nFree space or move output / --temp-dir / --work-dir to another filesystem. "
            "For AlphaQ, put --alphaq-cache on another disk, or explicitly use "
            "--alphaq-consume-fitted (discarding fitted-cache resume). "
            "Estimates include 5% + 64 MiB headroom per filesystem for codec/metadata/sidecar "
            "variation; concurrent writers and filesystem quotas can still change availability.",
        )


def source_asset_bytes(root: Path) -> int:
    """Budget local tokenizer/config sidecars without reading checkpoint payloads."""

    if not root.is_dir():
        return 0
    return sum(
        path.stat().st_size for path in root.iterdir()
        if path.is_file() and path.suffix in {".json", ".jinja", ".model", ".txt", ".tiktoken"}
    )
