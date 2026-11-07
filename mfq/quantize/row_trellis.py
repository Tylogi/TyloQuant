"""Experimental per-row Trellis container; not a registered MFQ runtime format."""

from dataclasses import dataclass, field
import operator
import struct

import numpy as np

GAINS = np.exp2(np.linspace(-1, 1, 16)).astype(np.float16)
CONSTANTS = {
    "1": {"gain": 1.05, "step": 1.0},
    "2": {"gain": 0.95, "step": 1.0},
    "3": {"gain": 0.85, "step": 1.0},
    "4": {"gain": 0.95, "step": 0.64},
    "5": {"gain": 0.95, "step": 0.4266666666666667},
    "6": {"gain": 0.95, "step": 0.3657142857142857},
    "7": {"gain": 0.95, "step": 0.17066666666666666},
    "8": {"gain": 0.95, "step": 0.08258064516129032},
}
_HEADER = struct.Struct("<8sIIIIII")
_MAGIC = b"UNTQ0002"


def validate_shape(shape, group_size, state_bits):
    if len(shape) != 2:
        raise ValueError("weight must be a nonempty 2-D matrix")
    rows, columns = map(operator.index, shape)
    if rows <= 0 or columns <= 0 or max(rows, columns) >= 2**32:
        raise ValueError("matrix dimensions must be positive uint32 values")
    if group_size not in (128, 256) or columns % group_size:
        raise ValueError("columns must be divisible by group_size (128 or 256)")
    if state_bits not in (10, 16):
        raise ValueError("state_bits must be 10 or 16")
    return rows, columns


def fixed_nbytes(shape, group_size=256, column_scaling=True):
    rows, columns = validate_shape(shape, group_size, 16)
    gain_words = (rows * (columns // group_size) + 7) // 8
    return 32 + rows * 3 + ((rows + 31) // 32) * 8 + gain_words * 4 + (
        columns // 2 if column_scaling else 0
    )


def allocate_bits(errors, max_q_sum):
    """Minimize total row SSE under an integer sum-of-q budget."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 8 or not errors.shape[0]:
        raise ValueError("errors must have shape (rows, 8)")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("errors must be finite and nonnegative")
    rows = errors.shape[0]
    available = min(operator.index(max_q_sum) - rows, 7 * rows)
    if available < 0:
        raise ValueError("byte budget cannot fit q1 and container metadata")
    costs = np.full(available + 1, np.inf, np.float64)
    costs[0] = 0.0
    history = np.zeros((rows, available + 1), np.uint8)
    for row in range(rows):
        updated = np.full_like(costs, np.inf)
        for extra in range(min(7, available) + 1):
            candidate = costs[:available + 1 - extra] + errors[row, extra]
            target = updated[extra:]
            improved = candidate < target
            target[improved] = candidate[improved]
            history[row, extra:][improved] = extra
        costs = updated
    remaining = int(np.argmin(costs))
    qs = np.empty(rows, np.uint8)
    for row in range(rows - 1, -1, -1):
        extra = int(history[row, remaining])
        qs[row] = extra + 1
        remaining -= extra
    return qs


def _integer_array(value, dtype, name):
    source = np.asarray(value)
    target = np.dtype(dtype)
    if source.dtype.kind not in "ui" or np.any(source < 0) or np.any(source > np.iinfo(target).max):
        raise ValueError(f"{name} must contain unsigned {target.itemsize * 8}-bit integers")
    result = source.astype(target, copy=True)
    result.flags.writeable = False
    return result

@dataclass(frozen=True)
class RowTrellisTensor:
    columns: int
    group_size: int
    state_bits: int
    descriptors: np.ndarray
    blocks: np.ndarray
    payload: np.ndarray
    column_codes: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))

    def __post_init__(self):
        descriptors = _integer_array(self.descriptors, "u1", "descriptors")
        if descriptors.ndim != 2 or descriptors.shape[1] != 3:
            raise ValueError("descriptors must have shape (rows, 3)")
        rows, columns = validate_shape(
            (descriptors.shape[0], self.columns), self.group_size, self.state_bits
        )
        blocks = _integer_array(self.blocks, "<u8", "blocks")
        payload = _integer_array(self.payload, "<u4", "payload")
        codes = _integer_array(self.column_codes, "u1", "column_codes")
        if blocks.shape != ((rows + 31) // 32,) or payload.ndim != 1:
            raise ValueError("invalid block table or payload shape")
        if codes.shape not in ((0,), (columns,)) or np.any(codes > 15):
            raise ValueError("column_codes must be empty or contain one 4-bit code per column")
        descriptor = (
            descriptors[:, 0].astype(np.uint32)
            | (descriptors[:, 1].astype(np.uint32) << 8)
            | (descriptors[:, 2].astype(np.uint32) << 16)
        )
        if np.any(descriptor >> 23) or np.any((descriptor & 4095) > 3967):
            raise ValueError("invalid row scale or reserved descriptor bit")
        qs = ((descriptor >> 12) & 7) + 1
        offset = (rows * (columns // self.group_size) + 7) // 8
        prefix = 0
        for row in range(rows):
            if row % 32 == 0:
                if int(blocks[row // 32]) != offset:
                    raise ValueError("block offset does not match packed row sizes")
                prefix = 0
            if int((descriptor[row] >> 15) & 255) != prefix:
                raise ValueError("row precision prefix does not match its block")
            offset += int(qs[row]) * (columns // 32)
            prefix += int(qs[row])
        if payload.size != offset:
            raise ValueError("payload length does not match row precisions")
        for name, value in (
            ("descriptors", descriptors), ("blocks", blocks),
            ("payload", payload), ("column_codes", codes),
        ):
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return self.descriptors.shape[0], self.columns

    @property
    def precisions(self):
        return ((self.descriptors[:, 1] >> 4) & 7) + 1

    @property
    def nbytes(self):
        return 32 + self.descriptors.nbytes + self.blocks.nbytes + self.payload.nbytes + self.column_codes.size // 2

    @property
    def bpw(self):
        return self.nbytes * 8 / (self.shape[0] * self.columns)

    def to_bytes(self):
        header = _HEADER.pack(
            _MAGIC, *self.shape, self.group_size, self.state_bits,
            4 if self.column_codes.size else 0, 1,
        )
        columns = (
            (self.column_codes[::2] | (self.column_codes[1::2] << 4)).tobytes()
            if self.column_codes.size else b""
        )
        return header + self.descriptors.tobytes() + self.blocks.tobytes() + columns + self.payload.tobytes()

    @classmethod
    def from_bytes(cls, blob):
        view = memoryview(blob)
        if view.nbytes < _HEADER.size:
            raise ValueError("truncated Trellis header")
        magic, rows, columns, group_size, state_bits, column_bits, version = _HEADER.unpack_from(view)
        if magic != _MAGIC or version != 1 or column_bits not in (0, 4):
            raise ValueError("unsupported Trellis container or constants version")
        validate_shape((rows, columns), group_size, state_bits)
        cursor = 32 + rows * 3 + ((rows + 31) // 32) * 8 + (columns // 2 if column_bits else 0)
        if view.nbytes < cursor or (view.nbytes - cursor) % 4:
            raise ValueError("truncated or misaligned Trellis payload")
        descriptors = np.frombuffer(view, "u1", rows * 3, 32).reshape(rows, 3)
        offset = 32 + rows * 3
        blocks = np.frombuffer(view, "<u8", (rows + 31) // 32, offset)
        offset += blocks.nbytes
        codes = np.empty(0, np.uint8)
        if column_bits:
            packed = np.frombuffer(view, "u1", columns // 2, offset)
            codes = np.stack((packed & 15, packed >> 4), axis=1).reshape(-1)
        payload = np.frombuffer(view, "<u4", offset=cursor)
        return cls(columns, group_size, state_bits, descriptors, blocks, payload, codes)


def assemble_rows(buckets, qs, shape, *, group_size=256, state_bits=16, column_codes=None):
    rows, columns = validate_shape(shape, group_size, state_bits)
    qs = np.asarray(qs)
    if qs.shape != (rows,) or qs.dtype.kind not in "ui" or np.any((qs < 1) | (qs > 8)):
        raise ValueError("one integer precision in [1, 8] is required per row")
    group_count = columns // group_size
    gains = np.zeros((rows * group_count + 7) // 8 * 8, np.uint32)
    descriptors = np.zeros(rows, np.uint32)
    blocks, chunks = [], []
    offset = gains.size // 8
    prefix = 0
    for row in range(rows):
        packed, anchor, gain = buckets[row]
        packed = _integer_array(packed, "<u4", "row payload")
        gain = _integer_array(gain, "u1", "group gains")
        if packed.shape != (columns * int(qs[row]) // 32,) or gain.shape != (group_count,) or np.any(gain > 15):
            raise ValueError("row payload or group gains do not match the matrix")
        anchor = np.asarray(anchor, dtype=np.float16)
        if anchor.ndim or not np.isfinite(anchor) or anchor < 0:
            raise ValueError("row anchor must be a finite nonnegative scalar")
        raw_anchor = int(anchor.view(np.uint16))
        if raw_anchor & 7:
            raise ValueError("row anchor is not representable by FP12")
        if row % 32 == 0:
            blocks.append(offset)
            prefix = 0
        descriptors[row] = (raw_anchor >> 3) | ((int(qs[row]) - 1) << 12) | (prefix << 15)
        chunks.append(packed)
        gains[row * group_count:(row + 1) * group_count] = gain
        offset += packed.size
        prefix += int(qs[row])
    gain_words = np.bitwise_or.reduce(
        gains.reshape(-1, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=1
    )
    descriptors = descriptors.astype("<u4").view(np.uint8).reshape(rows, 4)[:, :3].copy()
    return RowTrellisTensor(
        columns, group_size, state_bits, descriptors, np.array(blocks, np.uint64),
        np.concatenate([gain_words, *chunks]),
        np.empty(0, np.uint8) if column_codes is None else column_codes,
    )


def decode_cpu(tensor):
    """Independent NumPy decoder for container inspection and correctness tests."""
    rows, columns = tensor.shape
    result = np.empty(tensor.shape, np.float32)
    groups = columns // tensor.group_size
    column_factors = GAINS[tensor.column_codes].astype(np.float32) if tensor.column_codes.size else np.ones(columns, np.float32)
    for row in range(rows):
        d = int.from_bytes(tensor.descriptors[row].tobytes(), "little")
        q = ((d >> 12) & 7) + 1
        base_bits = min(q, 3)
        refine = q - base_bits
        anchor = np.array((d & 4095) << 3, np.uint16).view(np.float16).astype(np.float32)
        offset = int(tensor.blocks[row // 32]) + ((d >> 15) & 255) * (columns // 32)
        for group in range(groups):
            gain_index = row * groups + group
            gain = (int(tensor.payload[gain_index // 8]) >> (gain_index % 8 * 4)) & 15
            word_count = tensor.group_size * q // 32
            words = tensor.payload[offset + group * word_count:offset + (group + 1) * word_count]
            base_words = tensor.group_size * base_bits // 32
            bits = np.unpackbits(words[:base_words].astype(">u4").view(np.uint8))
            start = ((np.arange(tensor.group_size) + 1) * base_bits - tensor.state_bits) % bits.size
            state = np.zeros(tensor.group_size, np.uint32)
            for bit in range(tensor.state_bits):
                state = (state << 1) | bits[(start + bit) % bits.size]
            raw = state * np.uint32(0x83dcd12d)
            total = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24))
            accumulator = (total + 0x6400).astype(np.uint16).view(np.float16).astype(np.float32)
            inverse = np.array(0x1eee, np.uint16).view(np.float16).astype(np.float32)
            bias = np.array(0xc931, np.uint16).view(np.float16).astype(np.float32)
            decoded = (accumulator * inverse + bias).astype(np.float16).astype(np.float32)
            if refine:
                residual = np.unpackbits(words[base_words:].astype(">u4").view(np.uint8)).reshape(-1, refine)
                code = residual @ (1 << np.arange(refine - 1, -1, -1))
                decoded = ((code - ((1 << refine) - 1) * 0.5) * CONSTANTS[str(q)]["step"] + decoded).astype(np.float32)
            first = group * tensor.group_size
            result[row, first:first + tensor.group_size] = decoded * (anchor * np.float32(GAINS[gain])) * column_factors[first:first + tensor.group_size]
    return result
