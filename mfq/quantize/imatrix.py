"""Load and resolve llama.cpp importance matrices."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ImportanceEntry:
    values: np.ndarray
    counts: np.ndarray

    @property
    def matrices(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class ImportanceMatrix:
    path: Path
    entries: dict[str, ImportanceEntry]
    datasets: tuple[str, ...]
    chunk_count: int
    chunk_size: int
    legacy: bool

    def find(self, names: Iterable[str]) -> tuple[str, ImportanceEntry] | None:
        for name in names:
            entry = self.entries.get(name)
            if entry is not None:
                return name, entry
        return None

    def for_rows(
        self,
        names: Iterable[str],
        original_shape: tuple[int, ...],
        storage_shape: tuple[int, int],
        rows: slice | np.ndarray,
    ) -> tuple[str, np.ndarray] | None:
        match = self.find(names)
        if match is None:
            return None
        name, entry = match
        neuron_len = int(storage_shape[1])
        if entry.width != neuron_len:
            raise ValueError(
                f"imatrix width mismatch for {name}: {entry.width} != {neuron_len}"
            )
        if entry.matrices == 1:
            return name, entry.values[0]
        if len(original_shape) != 3:
            raise ValueError(
                f"imatrix for non-expert tensor {name} has {entry.matrices} matrices"
            )
        experts, rows_per_expert, _ = original_shape
        if entry.matrices != experts:
            raise ValueError(
                f"imatrix expert count mismatch for {name}: "
                f"{entry.matrices} != {experts}"
            )
        if isinstance(rows, slice):
            start = 0 if rows.start is None else int(rows.start)
            stop = int(storage_shape[0]) if rows.stop is None else int(rows.stop)
            row_ids = np.arange(start, stop, dtype=np.int64)
        else:
            row_ids = np.asarray(rows, dtype=np.int64).reshape(-1)
        if row_ids.size and (
            int(row_ids.min()) < 0 or int(row_ids.max()) >= int(storage_shape[0])
        ):
            raise IndexError(f"imatrix row selection is outside {storage_shape[0]} rows")
        expert_ids = row_ids // int(rows_per_expert)
        return name, np.ascontiguousarray(entry.values[expert_ids], dtype=np.float32)


def _load_gguf_reader():
    try:
        from gguf import GGUFReader  # type: ignore
    except ModuleNotFoundError:
        import sys

        gguf_py = Path(__file__).resolve().parents[2] / "references" / "llamacpp" / "gguf-py"
        if not gguf_py.exists():
            raise
        sys.path.insert(0, str(gguf_py))
        from gguf import GGUFReader  # type: ignore
    return GGUFReader


def _field_value(reader: Any, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    value = field.contents()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_entry(
    name: str,
    sums: np.ndarray,
    counts: np.ndarray,
) -> ImportanceEntry:
    sums = np.asarray(sums, dtype=np.float32).reshape(-1)
    raw_counts = np.asarray(counts, dtype=np.float32).reshape(-1)
    if not sums.size or not raw_counts.size or sums.size % raw_counts.size:
        raise ValueError(
            f"invalid imatrix entry {name}: {sums.size} sums for {raw_counts.size} counts"
        )
    if not np.isfinite(raw_counts).all() or np.any(raw_counts < 0):
        raise ValueError(f"imatrix entry {name} has invalid counts")
    counts = np.rint(raw_counts).astype(np.int64)
    width = sums.size // counts.size
    values = sums.reshape(counts.size, width).copy()
    positive = counts > 0
    values[positive] /= counts[positive, None].astype(np.float32)
    values[~positive] = 1.0
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"imatrix entry {name} contains invalid importance values")
    return ImportanceEntry(np.ascontiguousarray(values), counts)


def _load_gguf(path: Path) -> ImportanceMatrix:
    reader = _load_gguf_reader()(str(path), "r")
    general_type = str(_field_value(reader, "general.type", ""))
    if general_type and general_type != "imatrix":
        raise ValueError(f"GGUF file is not an imatrix: general.type={general_type!r}")
    metadata_keys = (
        "imatrix.datasets",
        "imatrix.chunk_count",
        "imatrix.chunk_size",
    )
    missing_metadata = [key for key in metadata_keys if key not in reader.fields]
    if missing_metadata:
        raise ValueError(
            f"GGUF imatrix is missing metadata {missing_metadata}: {path}"
        )
    datasets_value = _field_value(reader, "imatrix.datasets", [])
    datasets = tuple(datasets_value if isinstance(datasets_value, list) else [str(datasets_value)])
    chunk_count = int(_field_value(reader, "imatrix.chunk_count", 0))
    chunk_size = int(_field_value(reader, "imatrix.chunk_size", 0))

    tensors = {str(tensor.name): tensor for tensor in reader.tensors}
    bases: set[str] = set()
    for name in tensors:
        if name.endswith(".in_sum2"):
            bases.add(name[: -len(".in_sum2")])
        elif name.endswith(".counts"):
            bases.add(name[: -len(".counts")])
    entries: dict[str, ImportanceEntry] = {}
    for name in sorted(bases):
        sums_tensor = tensors.get(name + ".in_sum2")
        counts_tensor = tensors.get(name + ".counts")
        if sums_tensor is None or counts_tensor is None:
            raise ValueError(f"imatrix has mismatched sums/counts tensors for {name}")
        if str(sums_tensor.tensor_type.name) != "F32" or str(counts_tensor.tensor_type.name) != "F32":
            raise ValueError(f"imatrix tensors for {name} must be F32")
        entries[name] = _normalize_entry(name, sums_tensor.data, counts_tensor.data)
    if not entries:
        raise ValueError(f"imatrix contains no entries: {path}")
    return ImportanceMatrix(path, entries, datasets, chunk_count, chunk_size, False)


def _read_exact(handle, count: int, label: str) -> bytes:
    value = handle.read(count)
    if len(value) != count:
        raise ValueError(f"truncated legacy imatrix while reading {label}")
    return value


def _load_legacy(path: Path) -> ImportanceMatrix:
    entries: dict[str, ImportanceEntry] = {}
    datasets: tuple[str, ...] = ()
    chunk_count = 0
    with path.open("rb") as handle:
        (entry_count,) = struct.unpack("<i", _read_exact(handle, 4, "entry count"))
        if entry_count < 1:
            raise ValueError(f"legacy imatrix contains no entries: {path}")
        for index in range(entry_count):
            (name_len,) = struct.unpack("<i", _read_exact(handle, 4, "name length"))
            if name_len <= 0 or name_len > 1 << 20:
                raise ValueError(f"invalid legacy imatrix name length: {name_len}")
            name = _read_exact(handle, name_len, "entry name").decode("utf-8")
            ncall, nval = struct.unpack("<ii", _read_exact(handle, 8, "entry header"))
            if nval < 1:
                raise ValueError(f"legacy imatrix entry {name} has no values")
            sums = np.frombuffer(
                _read_exact(handle, nval * 4, "entry values"), dtype="<f4"
            ).astype(np.float32, copy=True)
            divisor = float(ncall) if ncall > 0 else 1.0
            values = sums / divisor
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"legacy imatrix entry {name} contains invalid values")
            entries[name] = ImportanceEntry(
                np.ascontiguousarray(values.reshape(1, -1)),
                np.asarray([ncall], dtype=np.int64),
            )
        tail = handle.read(4)
        if tail:
            if len(tail) != 4:
                raise ValueError("truncated legacy imatrix chunk count")
            (chunk_count,) = struct.unpack("<i", tail)
            dataset_len_raw = handle.read(4)
            if dataset_len_raw:
                if len(dataset_len_raw) != 4:
                    raise ValueError("truncated legacy imatrix dataset length")
                (dataset_len,) = struct.unpack("<i", dataset_len_raw)
                if dataset_len > 0:
                    datasets = (_read_exact(handle, dataset_len, "dataset").decode("utf-8"),)
    return ImportanceMatrix(path, entries, datasets, chunk_count, 0, True)


def load_importance_matrix(path: str | Path) -> ImportanceMatrix:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"imatrix file does not exist: {resolved}")
    with resolved.open("rb") as handle:
        magic = handle.read(4)
    return _load_gguf(resolved) if magic == b"GGUF" else _load_legacy(resolved)
