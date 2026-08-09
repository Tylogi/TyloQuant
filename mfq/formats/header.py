"""MFQ file headers and tensor-level metadata.

An MFQ file consists of one global header and several tensor records. The header stores the magic number,
version, and model-structure summary. Each tensor record stores its precision specification (NINT variant,
group size, and so on) and the location of its compressed weight blob.

See :mod:`mfq.formats.io` for the binary layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MFQ_MAGIC = b"MFQ1"


@dataclass
class FileHeader:
    """Global MFQ file header."""

    magic: bytes = MFQ_MAGIC
    version: int = 1
    model_arch: str = ""
    num_tensors: int = 0
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class TensorRecord:
    """Location and precision metadata for one tensor in the file."""

    name: str
    dtype: str          # "NINT4" / "NINT5" ...
    shape: tuple[int, ...]
    offset: int         # Byte offset of the weight blob in the file
    nbytes: int         # Size of the weight blob in bytes
    groupsize: int = 0
    hadamard: bool = False
