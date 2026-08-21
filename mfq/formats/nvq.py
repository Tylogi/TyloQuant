"""NVQ: neuron-anchored importance-aware lattice quantization.

NVQ keeps the integer magnitude grids and even-parity sign coding used by
llama.cpp IQ2/IQ3, then replaces the 256-weight super-block scale with one
FP16 anchor per output neuron and a packed relative scale per small group::

    w[j, i] ~= d[j] * s[j, i // group_size] * c[index[j, i]] * sign[j, i]

The first two profiles intentionally use the original 256-entry E8/D4 grids:

``NVQ2_E8``
    One 8-D E8 magnitude index (8 bits) and one even-parity sign mask (7 bits)
    per eight weights.

``NVQ3_D4``
    One 4-D D4 magnitude index (8 bits) per four weights and one even-parity
    sign mask (7 bits) per eight weights.

``NVQ3_D4_512``
    One 4-D D4 magnitude index (9 bits) per four weights.  This retains the
    512-entry lattice used by llama.cpp IQ3_S while keeping NVQ's parity-sign
    and neuron-anchor layout.

``NVQ2_E8_1024`` / ``NVQ2_E8_4096``
    Ten- and twelve-bit E8 indices.  With the common JSC state and parity
    streams their asymptotic information payloads are 2.2917 and 2.5417 bpw.
    NVQ2J-XL may instead store each 24-weight group in one aligned 64-bit
    execution record (2.6667 bpw) to avoid a duplicate runtime cache.

``NVQ3_D4_1024``
    One 10-bit D4 index per four weights.  With JSC state and parity streams
    its asymptotic payload is 3.5417 bpw.

``NVQ2J`` / ``NVQ3J``
    One learned 4-bit joint scale/codebook state per 24 weights.  The state
    selects both an FP16 relative scale and one of 1/2/4 raw-int8 codebook
    banks.  Vector indices and parity signs retain the NVQ2 E8 or NVQ3 D4
    base layout selected by the tensor.

The embedded tables are byte-for-byte copies of ``kgrid_2bit_256`` and
``kgrid_256`` from llama.cpp ``ggml-quants.c`` at commit
``c264f65ff9d8f592a590e3221f712a8883b7dd81``. Their SHA-256 hashes are checked at import time so a damaged
table cannot silently change the format.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np


_E8_256_HEX = """
00000200050008000a00110014002000220028002a00410044005000580061006400800082008a00a20001010401100115014001840198010002020222028202
010404041004210424044004420448046004810484049004a404000502050805200546056905800591050906100640068406a406000805080808140828084108
440850085208880804094009020a140a01100410101021104010601084109010951000110811201150115a118011241245120014081420142514491480141815
6215001616160118041810184018811800190519a019511a002002200a2044206120802082202921482100220222012404241024402456240025412564259026
082820289428442a0140044010401840214024404040484056406040814084409040004120416141804185410142104248425642684200440844204480449944
124524450046014804481048404845480049584961498249454a904a005008501150195020508050885004514251a4519152905492540a550156545600581158
195864584059085a04601060406068600061556118626062006405641065126584654268008002800a8041808280048118814081118201840484108415844084
608400854685948509864086608602880489118a0490109024904090a19016918091459200942294449451958198209902a050a085a009a100a218a450a804a9
"""

_D4_256_HEX = """
00000200040009000b000f0010001200190022003b003d004100430048004a005100550058005a0061006c00780080008200840089009000920099009b009f00
a900af00bd00c100c700c800ca00d500f8000b011f0124012f013b013d01410147015a016a019d01b401c801cc01ce01e301f1010102030208020a0211021302
18021a021c0227022802400242024902500252028102830288028a0291029802ba02c002c202d002d902e602f60201030503280350035403660379038503d203
e0030004020409040b041004120416041904220441044304450448044a04510458047304770478048004820489048f04900492049f04a004ad04c104c804cc04
f804fc041d052b054305570561057c05c105c305ce05e505010608060a0611061306280635063a064006420650065906640666068106830688069506aa06ba06
c906db06180727073a074007460752076d078c079e07b307db07f00704080f081d081f082b082f087c0890089f08a008b008b608c708e5080409290934095509
63097809c509c809ca09d8090a0a210a380a400a460a560a6d0a8c0a9a0aba0ac20aeb0a080b130b170b3a0b420b590ba80bd40be20b140c240c260c340c510c
710c8f0cb40cd80cde0c240d450d6a0d9b0dc30dd10d030e050e070e080e1a0e2a0e560e600e8a0ea50eaa0ec00ecd0edb0ef00e110f210f400f420f540f980f
"""

_D4_512_HEX = """
00000100020005000700080009000a000c000e001000110015001b00200022002500270029002b003000320039003c003f004000410042004400480049004d00
50005300570059005d006400710075007a00800081008500870088008b008e009100950098009c00a200a500a700a900ab00b800bb00c300c900cd00d000d200
d900db00de00e400e800ea00f700f900fd0000010b010f01110114011a012001230129013801420144015001520156015b0161016501670176017b0186018901
8b019901aa01b901c001c201c401d001d201d601db01e801ec010002010202020402080209020b020d020f021002120219021c021e022c022e0231023a024002
41024302460248024c02510258025b02610268026a0278027e0280028a028d028f02900294029a02a002a302ad02b002ba02c102c402c702c802cb02d102d702
d802dc02e102f202f802030305030a030c0319031b032203260328032c03410348034b03510358035a036903900394039703a403a603c103c303c803ca03d103
dd03e103f203f8030004010403040504070408040a040c040e0411041304170418041a0421042304250428042a04370438043b043d044004420448044b044f04
5204550459045c04620469048104840487049104930498049f04a104ab04af04b904bc04c004c204c504c904d204d404d904db04e204e804f604010507051005
13051a051d0521053a053d054405490552055f0560056b0578058005820587059105ad05b105c505c905d605d805e305e805020609060b060d060f0612061906
1b061d062206240627062906330639064106430645064c0651065306600670067a067e0680068a0690069a069c06a806c106c806cc06d106d306d706d806e106
e306e706ed06fb06030709070e07120719072a0730073407410748074a0755075a0764076e077b078e07a107c107c307d007d207dd07000804080a0810081408
17081a082108280838084208470849084b08520858085d08630869086d088108830888088d0891089b08a008af08b208b808c408c908cb08d008d208d908dd08
0009020914091f092009390945094709510972098b099c09b009c809cd09d809e309e909020a080a0c0a120a200a240a270a2a0a360a3c0a410a430a450a4a0a
510a5a0a7a0a800a890a930a980a9e0aab0ac20ac70ac80ad70ae40ae90af50afb0a010b040b100b1a0b260b4a0b560b690b6b0ba20bc20bc40bd20b090c0b0c
0d0c190c1b0c300c400c500c570c740c8a0c9c0ca20cad0cb20cb80cc00ccc0cd10ce00c150d230d320d400d430d5c0d700d850da00dc90dcb0d000e040e070e
100e120e1e0e200e2c0e320e420e490e540e630e650e810e840e880e8e0e910e980ea90ec20eda0edd0eeb0e010f050f0b0f100f280f520f620f820f990fc00f
"""

_TABLE_HASHES = {
    "e8_256": "915952ebc4af8ddba8113583ca1316fc818843f7b825bbf1a25ae1c39bf0e16d",
    "d4_256": "c9678803c61b2210fbd8cbbb3f965a835409cd7d88a04d086d236261bd221d11",
    "d4_512": "353274c58dd7d8dc5ee960dc8fc54ff9164eaf2b49206b841cd49415cf393f81",
}


def _decode_grid(hex_text: str, *, dims: int, digit_bits: int, expected_hash: str) -> np.ndarray:
    raw = bytes.fromhex(hex_text)
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise RuntimeError("NVQ codebook checksum mismatch")
    encoded = np.frombuffer(raw, dtype="<u2")
    shifts = digit_bits * np.arange(dims, dtype=np.uint16)
    digits = (encoded[:, None] >> shifts[None, :]) & ((1 << digit_bits) - 1)
    return np.ascontiguousarray(2 * digits + 1, dtype=np.int8)


E8_256 = _decode_grid(
    _E8_256_HEX,
    dims=8,
    digit_bits=2,
    expected_hash=_TABLE_HASHES["e8_256"],
)
D4_256 = _decode_grid(
    _D4_256_HEX,
    dims=4,
    digit_bits=3,
    expected_hash=_TABLE_HASHES["d4_256"],
)
D4_512 = _decode_grid(
    _D4_512_HEX,
    dims=4,
    digit_bits=3,
    expected_hash=_TABLE_HASHES["d4_512"],
)


@dataclass(frozen=True)
class NvqSpec:
    """Storage and codebook parameters for one NVQ tensor."""

    codebook: str
    groupsize: int = 24
    sub_bits: int = 4
    sign_mode: str = "even"

    def __post_init__(self) -> None:
        if self.codebook not in {
            "e8_256",
            "e8_1024",
            "e8_4096",
            "d4_256",
            "d4_512",
            "d4_1024",
        }:
            raise ValueError(f"unsupported NVQ codebook: {self.codebook}")
        if self.groupsize <= 0 or self.groupsize % 8:
            raise ValueError("NVQ groupsize must be a positive multiple of 8")
        if not 1 <= self.sub_bits <= 8:
            raise ValueError("NVQ sub_bits must be in [1, 8]")
        if self.sign_mode not in {"even", "index_parity"}:
            raise ValueError(f"unsupported NVQ sign mode: {self.sign_mode}")
        if self.sign_mode == "index_parity" and self.codebook != "e8_256":
            raise ValueError("index_parity currently requires the 8-D E8 codebook")

    @property
    def vector_size(self) -> int:
        return 8 if self.codebook.startswith("e8_") else 4

    @property
    def index_bits(self) -> int:
        return {
            "e8_256": 8,
            "e8_1024": 10,
            "e8_4096": 12,
            "d4_256": 8,
            "d4_512": 9,
            "d4_1024": 10,
        }[self.codebook]

    @property
    def codebook_entries(self) -> int:
        return 1 << self.index_bits

    @property
    def base_bpw(self) -> float:
        return self.index_bits / self.vector_size + 7.0 / 8.0

    @property
    def label(self) -> str:
        base = {
            "e8_256": "NVQ2-E8-256",
            "e8_1024": "NVQ2-E8-1024",
            "e8_4096": "NVQ2-E8-4096",
            "d4_256": "NVQ3-D4-256",
            "d4_512": "NVQ3-D4-512",
            "d4_1024": "NVQ3-D4-1024",
        }[self.codebook]
        return f"{base}-IP" if self.sign_mode == "index_parity" else base

    def payload_nbytes(self, out: int, neuron_len: int) -> int:
        """Packed tensor bytes excluding the self-describing blob header."""

        ng = math.ceil(neuron_len / self.groupsize)
        nvec = math.ceil(neuron_len / self.vector_size)
        nsign = math.ceil(neuron_len / 8)
        anchors = out * 2
        scales = (out * ng * self.sub_bits + 7) // 8
        indices = (out * nvec * self.index_bits + 7) // 8
        signs = (out * nsign * 7 + 7) // 8
        return anchors + scales + indices + signs

    def bpw(self, neuron_len: int, *, out: int = 1) -> float:
        """Actual packed payload bpw, including tail groups and byte rounding."""

        return 8.0 * self.payload_nbytes(out, neuron_len) / (out * neuron_len)


NVQ2_E8 = NvqSpec("e8_256", groupsize=24, sub_bits=4)
NVQ2_E8_1024 = NvqSpec("e8_1024", groupsize=24, sub_bits=4)
NVQ2_E8_4096 = NvqSpec("e8_4096", groupsize=24, sub_bits=4)
NVQ3_D4 = NvqSpec("d4_256", groupsize=24, sub_bits=4)
NVQ3_D4_512 = NvqSpec("d4_512", groupsize=24, sub_bits=4)
NVQ3_D4_1024 = NvqSpec("d4_1024", groupsize=24, sub_bits=4)


def _bit_reverse(values: np.ndarray, bits: int) -> np.ndarray:
    result = np.zeros_like(values)
    work = values.copy()
    for _ in range(bits):
        result = (result << 1) | (work & 1)
        work >>= 1
    return result


def _extend_lattice_codebook(
    base: np.ndarray,
    *,
    dims: int,
    digit_bits: int,
    entries: int,
) -> np.ndarray:
    """Build a deterministic nested lattice codebook without stored tables."""

    full_entries = 1 << (dims * digit_bits)
    if entries > full_entries:
        raise ValueError("requested NVQ codebook exceeds the source lattice")
    shifts = digit_bits * np.arange(dims, dtype=np.uint32)
    base_digits = ((base.astype(np.uint32) - 1) // 2).astype(np.uint32)
    base_ids = np.bitwise_or.reduce(base_digits << shifts[None, :], axis=1)
    used = np.zeros(full_entries, dtype=np.bool_)
    used[base_ids] = True
    order = _bit_reverse(np.arange(full_entries, dtype=np.uint32), dims * digit_bits)
    extra_ids = order[~used[order]][: entries - base.shape[0]]
    selected = np.concatenate((base_ids, extra_ids))
    digits = (selected[:, None] >> shifts[None, :]) & ((1 << digit_bits) - 1)
    return np.ascontiguousarray(2 * digits + 1, dtype=np.int8)


E8_1024 = _extend_lattice_codebook(
    E8_256, dims=8, digit_bits=2, entries=1024
)
E8_4096 = _extend_lattice_codebook(
    E8_1024, dims=8, digit_bits=2, entries=4096
)
D4_1024 = _extend_lattice_codebook(
    D4_512, dims=4, digit_bits=3, entries=1024
)


def codebook_for(spec: NvqSpec) -> np.ndarray:
    return {
        "e8_256": E8_256,
        "e8_1024": E8_1024,
        "e8_4096": E8_4096,
        "d4_256": D4_256,
        "d4_512": D4_512,
        "d4_1024": D4_1024,
    }[spec.codebook]


def validate_codebook(spec: NvqSpec, codebook: np.ndarray) -> np.ndarray:
    """Validate one runtime codebook and return packed-domain int8 values."""

    value = np.asarray(codebook)
    expected = (spec.codebook_entries, spec.vector_size)
    if value.shape != expected:
        raise ValueError(f"NVQ codebook has shape {value.shape}, expected {expected}")
    rounded = np.rint(value)
    max_level = 7 if spec.vector_size == 8 else 15
    if (
        not np.array_equal(value, rounded)
        or np.any(rounded < 1)
        or np.any(rounded > max_level)
        or np.any((rounded.astype(np.int16) & 1) == 0)
    ):
        raise ValueError(f"NVQ codebook entries must be odd integers in [1, {max_level}]")
    result = np.ascontiguousarray(rounded, dtype=np.int8)
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("NVQ codebook entries must be unique")
    return result


def pack_codebook(spec: NvqSpec, codebook: np.ndarray) -> bytes:
    value = validate_codebook(spec, codebook)
    digits = ((value.astype(np.uint16) - 1) // 2).astype(np.uint16)
    digit_bits = 2 if spec.vector_size == 8 else 3
    shifts = (digit_bits * np.arange(spec.vector_size, dtype=np.uint16))[None, :]
    packed = np.bitwise_or.reduce(digits << shifts, axis=1).astype("<u2")
    return packed.tobytes()


def unpack_codebook(spec: NvqSpec, payload: bytes) -> np.ndarray:
    expected = spec.codebook_entries * 2
    if len(payload) != expected:
        raise ValueError(
            f"packed NVQ codebook must be {expected} bytes, got {len(payload)}"
        )
    packed = np.frombuffer(payload, dtype="<u2")
    digit_bits = 2 if spec.vector_size == 8 else 3
    shifts = (digit_bits * np.arange(spec.vector_size, dtype=np.uint16))[None, :]
    digits = (packed[:, None] >> shifts) & ((1 << digit_bits) - 1)
    return validate_codebook(spec, 2 * digits + 1)


@dataclass
class NvqTensor:
    """Packed-domain NVQ tensor before byte serialization."""

    spec: NvqSpec
    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    neuron_scale: np.ndarray
    sub_scale: np.ndarray
    indices: np.ndarray
    signs: np.ndarray
    codebook: np.ndarray | None = None

    @property
    def payload_nbytes(self) -> int:
        custom = self.spec.codebook_entries * 2 if self.codebook is not None else 0
        return custom + self.spec.payload_nbytes(self.neuron_scale.size, self.neuron_len)

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


_JSC_STATE_COUNT = 16
_JSC_METADATA_BYTES = 64
_JSC_VERSION = 1
_JSC_GROUP_LAYOUT_VERSION = 2
_JSC_ANALYTIC_STATE = 1
_JSC_GROUP64_LAYOUT = 1
_JSC_STREAM_LAYOUT = "streams"
_JSC_AUTO_LAYOUT = "auto"
_JSC_GROUP64_LAYOUT_NAME = "group64"


def _analytic_jsc_tables(
    banks: int,
    *,
    vector_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.arange(_JSC_STATE_COUNT, dtype=np.uint8)
    bank = state % banks
    if banks == 1:
        scale = state.astype(np.float32)
    elif vector_size == 4:
        scale = (state // banks).astype(np.float32) + 1.0
    else:
        rank = state // banks
        scale = 15.0 * (rank.astype(np.float32) + 1.0) / (16 // banks)
    return scale, bank


def validate_jsc_codebooks(
    codebooks: np.ndarray,
    *,
    vector_size: int | None = None,
    codebook_entries: int | None = None,
) -> np.ndarray:
    """Validate deployment NVQ-JSC codebook banks."""

    value = np.asarray(codebooks)
    valid_vector_size = value.ndim == 3 and value.shape[2] in {4, 8}
    valid_entries = value.ndim == 3 and value.shape[1] in {256, 512, 1024, 4096}
    if (
        value.ndim != 3
        or value.shape[0] not in {1, 2, 4}
        or not valid_entries
        or not valid_vector_size
        or (vector_size is not None and value.shape[2] != vector_size)
        or (
            codebook_entries is not None
            and value.shape[1] != codebook_entries
        )
    ):
        expected = vector_size if vector_size is not None else "4 or 8"
        entries = (
            codebook_entries
            if codebook_entries is not None
            else "256, 512, 1024, or 4096"
        )
        raise ValueError(
            f"NVQ-JSC codebooks must have shape [banks,{entries},{expected}] "
            "with banks in {1,2,4}"
        )
    if value.shape[1] == 512 and value.shape[2] != 4:
        raise ValueError("512-entry NVQ-JSC codebooks require 4-D vectors")
    if value.shape[1] == 4096 and value.shape[2] != 8:
        raise ValueError("4096-entry NVQ-JSC codebooks require 8-D vectors")
    rounded = np.rint(value)
    if (
        not np.isfinite(value).all()
        or not np.array_equal(value, rounded)
        or np.any(rounded < 0)
        or np.any(rounded > 127)
    ):
        raise ValueError("NVQ-JSC deployment codebooks must be integers in [0,127]")
    result = np.ascontiguousarray(rounded, dtype=np.int8)
    if np.any(np.all(result == 0, axis=-1)):
        raise ValueError("NVQ-JSC codewords must not be all zero")
    return result


@dataclass
class NvqJscTensor:
    """Production NVQ-JSC tensor with compact joint scale/codebook states."""

    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    neuron_scale: np.ndarray
    scale_lut: np.ndarray
    bank_for_state: np.ndarray
    state: np.ndarray
    indices: np.ndarray
    signs: np.ndarray
    codebooks: np.ndarray
    base_spec: NvqSpec = NVQ2_E8
    storage_layout: str = _JSC_AUTO_LAYOUT

    @property
    def spec(self) -> NvqSpec:
        return self.base_spec

    @property
    def sub_scale(self) -> np.ndarray:
        """Expose the state stream under the common NVQ runtime name."""

        return self.state

    @property
    def payload_nbytes(self) -> int:
        out = int(np.asarray(self.neuron_scale).size)
        tables = _JSC_METADATA_BYTES + int(np.asarray(self.codebooks).size)
        if _resolve_jsc_storage_layout(self) == _JSC_GROUP64_LAYOUT_NAME:
            groups = math.ceil(self.neuron_len / self.spec.groupsize)
            return tables + out * 2 + out * groups * 8
        return tables + self.spec.payload_nbytes(out, self.neuron_len)

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


def pack_jsc_tables(
    scale_lut: np.ndarray,
    bank_for_state: np.ndarray,
    codebooks: np.ndarray,
    *,
    storage_layout: str = _JSC_STREAM_LAYOUT,
) -> bytes:
    """Pack the fixed 64-byte JSC header and raw int8 codebook banks."""

    codebooks = validate_jsc_codebooks(codebooks)
    banks = codebooks.shape[0]
    scale_lut = np.asarray(scale_lut, dtype=np.float32).reshape(-1)
    bank_for_state = np.asarray(bank_for_state).reshape(-1)
    if scale_lut.shape != (_JSC_STATE_COUNT,):
        raise ValueError("NVQ-JSC scale_lut must contain 16 values")
    if not np.isfinite(scale_lut).all() or np.any(scale_lut < 0):
        raise ValueError("NVQ-JSC scale_lut must be finite and non-negative")
    if bank_for_state.shape != (_JSC_STATE_COUNT,):
        raise ValueError("NVQ-JSC bank_for_state must contain 16 values")
    if not np.issubdtype(bank_for_state.dtype, np.integer):
        raise ValueError("NVQ-JSC bank_for_state must contain integers")
    bank_for_state = np.ascontiguousarray(bank_for_state, dtype=np.uint8)
    if np.any(bank_for_state >= banks):
        raise ValueError("NVQ-JSC bank_for_state references a missing bank")

    if storage_layout not in {
        _JSC_STREAM_LAYOUT,
        _JSC_GROUP64_LAYOUT_NAME,
    }:
        raise ValueError(f"unsupported NVQ-JSC storage layout: {storage_layout}")
    if (
        storage_layout == _JSC_GROUP64_LAYOUT_NAME
        and codebooks.shape[1:] != (4096, 8)
    ):
        raise ValueError("NVQ-JSC group64 storage requires 4096-entry 8-D codebooks")
    header = bytearray(_JSC_METADATA_BYTES)
    header[0] = (
        _JSC_GROUP_LAYOUT_VERSION
        if storage_layout == _JSC_GROUP64_LAYOUT_NAME
        else _JSC_VERSION
    )
    header[1] = banks
    header[2] = _JSC_STATE_COUNT
    analytic_scale, analytic_bank = _analytic_jsc_tables(
        banks, vector_size=codebooks.shape[2]
    )
    if np.array_equal(scale_lut, analytic_scale) and np.array_equal(
        bank_for_state, analytic_bank
    ):
        header[3] = _JSC_ANALYTIC_STATE
    header[4:36] = np.ascontiguousarray(scale_lut, dtype="<f2").tobytes()
    header[36:52] = bank_for_state.tobytes()
    if storage_layout == _JSC_GROUP64_LAYOUT_NAME:
        header[52] = _JSC_GROUP64_LAYOUT
    return bytes(header) + codebooks.tobytes()


def pack_jsc_metadata(tensor: NvqJscTensor) -> bytes:
    return pack_jsc_tables(
        tensor.scale_lut,
        tensor.bank_for_state,
        tensor.codebooks,
        storage_layout=_resolve_jsc_storage_layout(tensor),
    )


def unpack_jsc_metadata(
    payload: bytes | memoryview,
    *,
    vector_size: int = 8,
    codebook_entries: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode JSC metadata and return ``(lut, bank_map, codebooks, bytes)``."""

    scale_lut, bank_for_state, codebooks, consumed, _ = (
        _unpack_jsc_metadata_profile(
            payload,
            vector_size=vector_size,
            codebook_entries=codebook_entries,
        )
    )
    return scale_lut, bank_for_state, codebooks, consumed


def _unpack_jsc_metadata_profile(
    payload: bytes | memoryview,
    *,
    vector_size: int = 8,
    codebook_entries: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Decode JSC tables plus their physical storage-layout contract."""

    if len(payload) < _JSC_METADATA_BYTES:
        raise ValueError("truncated NVQ-JSC metadata header")
    header = memoryview(payload)[:_JSC_METADATA_BYTES]
    version = int(header[0])
    banks = int(header[1])
    states = int(header[2])
    if version not in {_JSC_VERSION, _JSC_GROUP_LAYOUT_VERSION}:
        raise ValueError(f"unsupported NVQ-JSC metadata version: {version}")
    if banks not in {1, 2, 4} or states != _JSC_STATE_COUNT:
        raise ValueError(f"invalid NVQ-JSC table dimensions: banks={banks}, states={states}")
    state_mode = int(header[3])
    layout = _JSC_STREAM_LAYOUT
    if version == _JSC_GROUP_LAYOUT_VERSION:
        if int(header[52]) != _JSC_GROUP64_LAYOUT or any(header[53:64]):
            raise ValueError("invalid NVQ-JSC v2 storage layout")
        layout = _JSC_GROUP64_LAYOUT_NAME
    elif any(header[52:64]):
        raise ValueError("NVQ-JSC v1 reserved metadata bytes must be zero")
    if state_mode not in {0, _JSC_ANALYTIC_STATE}:
        raise ValueError("NVQ-JSC reserved metadata bytes must be zero")
    scale_lut = np.frombuffer(header[4:36], dtype="<f2").astype(np.float32)
    bank_for_state = np.frombuffer(header[36:52], dtype=np.uint8).copy()
    if vector_size not in {4, 8}:
        raise ValueError(f"invalid NVQ-JSC vector size: {vector_size}")
    if codebook_entries not in {256, 512, 1024, 4096}:
        raise ValueError(f"invalid NVQ-JSC codebook size: {codebook_entries}")
    if codebook_entries == 512 and vector_size != 4:
        raise ValueError("512-entry NVQ-JSC codebooks require 4-D vectors")
    if codebook_entries == 4096 and vector_size != 8:
        raise ValueError("4096-entry NVQ-JSC codebooks require 8-D vectors")
    codebook_bytes = banks * codebook_entries * vector_size
    consumed = _JSC_METADATA_BYTES + codebook_bytes
    if len(payload) < consumed:
        raise ValueError("truncated NVQ-JSC codebook banks")
    raw = np.frombuffer(
        memoryview(payload)[_JSC_METADATA_BYTES:consumed], dtype=np.int8
    ).copy().reshape(banks, codebook_entries, vector_size)
    codebooks = validate_jsc_codebooks(
        raw,
        vector_size=vector_size,
        codebook_entries=codebook_entries,
    )
    if not np.isfinite(scale_lut).all() or np.any(scale_lut < 0):
        raise ValueError("invalid NVQ-JSC scale LUT")
    if np.any(bank_for_state >= banks):
        raise ValueError("invalid NVQ-JSC state-to-bank map")
    if state_mode == _JSC_ANALYTIC_STATE:
        expected_scale, expected_bank = _analytic_jsc_tables(
            banks, vector_size=vector_size
        )
        if not np.array_equal(scale_lut, expected_scale) or not np.array_equal(
            bank_for_state, expected_bank
        ):
            raise ValueError("invalid analytic NVQ-JSC state tables")
    return scale_lut, bank_for_state, codebooks, consumed, layout


def _resolve_jsc_storage_layout(tensor: NvqJscTensor) -> str:
    return resolve_jsc_storage_layout(tensor.spec, tensor.storage_layout)


def resolve_jsc_storage_layout(
    spec: NvqSpec,
    storage_layout: str = _JSC_AUTO_LAYOUT,
) -> str:
    """Resolve and validate the physical JSC stream layout for ``spec``."""

    layout = storage_layout
    if layout == _JSC_AUTO_LAYOUT:
        layout = (
            _JSC_GROUP64_LAYOUT_NAME
            if spec.codebook == "e8_4096"
            else _JSC_STREAM_LAYOUT
        )
    if layout not in {_JSC_STREAM_LAYOUT, _JSC_GROUP64_LAYOUT_NAME}:
        raise ValueError(f"unsupported NVQ-JSC storage layout: {layout}")
    if layout == _JSC_GROUP64_LAYOUT_NAME and (
        spec.codebook != "e8_4096"
        or spec.vector_size != 8
        or spec.index_bits != 12
        or spec.groupsize != 24
        or spec.sub_bits != 4
    ):
        raise ValueError("NVQ-JSC group64 storage requires NVQ2J-XL")
    return layout


def jsc_payload_nbytes(
    spec: NvqSpec,
    out: int,
    neuron_len: int,
    *,
    storage_layout: str = _JSC_AUTO_LAYOUT,
) -> int:
    """Return anchor plus JSC stream bytes, excluding the table metadata."""

    layout = resolve_jsc_storage_layout(spec, storage_layout)
    if layout == _JSC_GROUP64_LAYOUT_NAME:
        return out * 2 + out * math.ceil(neuron_len / 24) * 8
    return spec.payload_nbytes(out, neuron_len)


def pack_jsc_group64(
    state: np.ndarray,
    indices: np.ndarray,
    signs: np.ndarray,
    *,
    neuron_len: int,
) -> bytes:
    """Pack NVQ2J-XL state, index, and sign rows as aligned 64-bit groups."""

    groups = math.ceil(neuron_len / 24)
    vectors = math.ceil(neuron_len / 8)
    state_values = np.asarray(state)
    if state_values.ndim != 2 or state_values.shape[1] != groups:
        raise ValueError(f"bad group64 state shape; expected [rows,{groups}]")
    out = int(state_values.shape[0])
    index_values = np.asarray(indices)
    sign_values = np.asarray(signs)
    if index_values.shape != (out, vectors):
        raise ValueError(f"bad group64 index shape; expected {(out, vectors)}")
    if sign_values.shape != (out, vectors):
        raise ValueError(f"bad group64 sign shape; expected {(out, vectors)}")
    for label, values, maximum in (
        ("state", state_values, 15),
        ("index", index_values, 4095),
        ("sign", sign_values, 127),
    ):
        if (
            not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or np.any(values > maximum)
        ):
            raise ValueError(f"group64 {label} values must be integers in [0,{maximum}]")
    states = state_values.astype(np.uint64, copy=False)
    indices = index_values.astype(np.uint64, copy=False)
    signs = sign_values.astype(np.uint8, copy=False)
    padded_indices = np.zeros((out, groups * 3), dtype=np.uint64)
    padded_signs = np.zeros((out, groups * 3), dtype=np.uint8)
    padded_indices[:, :vectors] = indices
    padded_signs[:, :vectors] = signs
    padded_indices = padded_indices.reshape(out, groups, 3)
    padded_signs = padded_signs.reshape(out, groups, 3)
    parity = np.bitwise_xor.reduce(
        (padded_signs[..., None] >> np.arange(7, dtype=np.uint8)) & 1,
        axis=-1,
    ).astype(np.uint64)
    sign8 = padded_signs.astype(np.uint64) | (parity << 7)
    records = states << np.uint64(60)
    for local in range(3):
        segment = padded_indices[..., local] | (sign8[..., local] << np.uint64(12))
        records |= segment << np.uint64(local * 20)
    return np.ascontiguousarray(records, dtype="<u8").tobytes()


def _pack_jsc_group64(tensor: NvqJscTensor) -> bytes:
    return pack_jsc_group64(
        tensor.state,
        tensor.indices,
        tensor.signs,
        neuron_len=tensor.neuron_len,
    )


def _unpack_jsc_group64(
    blob: bytes | memoryview,
    off: int,
    *,
    out: int,
    groups: int,
    vectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    count = out * groups
    nbytes = count * 8
    if off + nbytes > len(blob):
        raise ValueError("truncated NVQ-JSC group64 stream")
    records = np.frombuffer(blob, dtype="<u8", count=count, offset=off).copy()
    records = records.reshape(out, groups)
    state = (records >> np.uint64(60)).astype(np.uint8)
    indices = np.zeros((out, groups * 3), dtype=np.uint16)
    signs = np.zeros((out, groups * 3), dtype=np.uint8)
    parity_table = np.asarray([int(value).bit_count() & 1 for value in range(128)])
    for local in range(3):
        segment = (records >> np.uint64(local * 20)) & np.uint64((1 << 20) - 1)
        indices[:, local::3] = (segment & np.uint64(0xFFF)).astype(np.uint16)
        sign8 = (segment >> np.uint64(12)).astype(np.uint8)
        sign7 = sign8 & np.uint8(0x7F)
        if np.any((sign8 >> np.uint8(7)) != parity_table[sign7]):
            raise ValueError("invalid NVQ-JSC group64 parity bit")
        signs[:, local::3] = sign7
    if vectors < groups * 3 and (
        np.any(indices[:, vectors:]) or np.any(signs[:, vectors:])
    ):
        raise ValueError("NVQ-JSC group64 padding must be zero")
    return state, indices[:, :vectors], signs[:, :vectors], off + nbytes


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    arr = np.ascontiguousarray(values).reshape(-1)
    if bits == 8:
        return arr.astype(np.uint8, copy=False).tobytes()
    if not 1 <= bits <= 16:
        raise ValueError(f"unsupported packed width: {bits}")
    u = arr.astype(np.uint16, copy=False)
    if np.any(u > (1 << bits) - 1):
        raise ValueError(f"value exceeds {bits}-bit packed width")
    shifts = np.arange(bits, dtype=np.uint16)
    stream = ((u[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(stream.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(blob: bytes, off: int, count: int, bits: int) -> tuple[np.ndarray, int]:
    nbytes = (count * bits + 7) // 8
    if bits == 8:
        end = off + count
        return np.frombuffer(blob, dtype=np.uint8, count=count, offset=off).copy(), end
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint16)
    dtype = np.uint8 if bits <= 8 else np.uint16
    values = (stream.astype(np.uint16) * shifts).sum(axis=1).astype(dtype)
    return values, off + nbytes


_MAGIC = b"NVQ1"
_LEGACY_MAGIC = b"NIQ1"
_HEADER = struct.Struct("<4sBBHiiI")
_CODEBOOK_ID = {
    "e8_256": 1,
    "d4_256": 2,
    "d4_512": 3,
    "e8_1024": 4,
    "e8_4096": 5,
    "d4_1024": 6,
}
_ID_CODEBOOK = {v: k for k, v in _CODEBOOK_ID.items()}
_INDEX_PARITY_FLAG = 0x80
_CUSTOM_CODEBOOK_FLAG = 0x40
_JSC_FLAG = 0x20


def _nonnegative_f16_anchors(value: np.ndarray, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        anchors = np.ascontiguousarray(value, dtype=np.float16)
    if not np.isfinite(anchors).all() or np.signbit(anchors).any():
        raise ValueError(f"{label} neuron anchors must be finite and non-negative in FP16")
    return anchors


def pack_nvq(tensor: NvqTensor | NvqJscTensor) -> bytes:
    """Serialize NVQ metadata and all bit-packed streams."""

    if isinstance(tensor, NvqJscTensor):
        return pack_nvq_jsc(tensor)

    spec = tensor.spec
    out = tensor.neuron_scale.size
    ng = math.ceil(tensor.neuron_len / spec.groupsize)
    nvec = math.ceil(tensor.neuron_len / spec.vector_size)
    nsign = math.ceil(tensor.neuron_len / 8)
    if tensor.sub_scale.shape != (out, ng):
        raise ValueError(f"bad sub_scale shape: {tensor.sub_scale.shape}, expected {(out, ng)}")
    if tensor.indices.shape != (out, nvec):
        raise ValueError(f"bad indices shape: {tensor.indices.shape}, expected {(out, nvec)}")
    if tensor.signs.shape != (out, nsign):
        raise ValueError(f"bad signs shape: {tensor.signs.shape}, expected {(out, nsign)}")
    anchors = _nonnegative_f16_anchors(tensor.neuron_scale, "NVQ")

    custom_codebook = (
        pack_codebook(spec, tensor.codebook) if tensor.codebook is not None else b""
    )
    parts = [
        _HEADER.pack(
            _MAGIC,
            _CODEBOOK_ID[spec.codebook]
            | (_INDEX_PARITY_FLAG if spec.sign_mode == "index_parity" else 0)
            | (_CUSTOM_CODEBOOK_FLAG if tensor.codebook is not None else 0),
            spec.sub_bits,
            spec.groupsize,
            tensor.axis,
            tensor.neuron_len,
            len(tensor.shape),
        ),
        struct.pack(f"<{len(tensor.shape)}q", *tensor.shape),
        struct.pack("<I", out),
        custom_codebook,
        anchors.tobytes(),
        _pack_bits(tensor.sub_scale, spec.sub_bits),
        _pack_bits(tensor.indices, spec.index_bits),
        _pack_bits(tensor.signs, 7),
    ]
    return b"".join(parts)


def pack_nvq_jsc(tensor: NvqJscTensor) -> bytes:
    """Serialize the production NVQ-JSC profile."""

    out = int(np.asarray(tensor.neuron_scale).size)
    spec = tensor.spec
    if spec.groupsize != 24 or spec.sub_bits != 4 or spec.sign_mode != "even":
        raise ValueError("NVQ-JSC serialization requires gs24, 4-bit state, and parity signs")
    validate_jsc_codebooks(
        tensor.codebooks,
        vector_size=spec.vector_size,
        codebook_entries=spec.codebook_entries,
    )
    ng = math.ceil(tensor.neuron_len / spec.groupsize)
    nvec = math.ceil(tensor.neuron_len / spec.vector_size)
    nsign = math.ceil(tensor.neuron_len / 8)
    if tensor.axis != 0 or len(tensor.shape) != 2:
        raise ValueError("NVQ-JSC serialization requires a rank-2 axis=0 tensor")
    if tuple(tensor.shape) != (out, tensor.neuron_len):
        raise ValueError("NVQ-JSC shape does not match neuron anchors and neuron_len")
    if np.asarray(tensor.state).shape != (out, ng):
        raise ValueError(f"bad NVQ-JSC state shape; expected {(out, ng)}")
    if np.asarray(tensor.indices).shape != (out, nvec):
        raise ValueError(f"bad NVQ-JSC indices shape; expected {(out, nvec)}")
    if np.asarray(tensor.signs).shape != (out, nsign):
        raise ValueError(f"bad NVQ-JSC signs shape; expected {(out, nsign)}")
    state = np.asarray(tensor.state)
    if not np.issubdtype(state.dtype, np.integer) or np.any(state < 0) or np.any(state > 15):
        raise ValueError("NVQ-JSC states must be integers in [0,15]")
    indices = np.asarray(tensor.indices)
    signs = np.asarray(tensor.signs)
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= spec.codebook_entries)
    ):
        raise ValueError(
            f"NVQ-JSC indices must be integers in [0,{spec.codebook_entries - 1}]"
        )
    if not np.issubdtype(signs.dtype, np.integer) or np.any(signs < 0) or np.any(signs > 127):
        raise ValueError("NVQ-JSC signs must be integers in [0,127]")
    anchors = _nonnegative_f16_anchors(tensor.neuron_scale, "NVQ-JSC")
    storage_layout = _resolve_jsc_storage_layout(tensor)
    if storage_layout == _JSC_GROUP64_LAYOUT_NAME:
        packed_streams = [_pack_jsc_group64(tensor)]
    else:
        packed_streams = [
            _pack_bits(state, 4),
            _pack_bits(indices, spec.index_bits),
            _pack_bits(signs, 7),
        ]

    return b"".join(
        [
            _HEADER.pack(
                _MAGIC,
                _CODEBOOK_ID[spec.codebook] | _JSC_FLAG,
                4,
                24,
                tensor.axis,
                tensor.neuron_len,
                len(tensor.shape),
            ),
            struct.pack(f"<{len(tensor.shape)}q", *tensor.shape),
            struct.pack("<I", out),
            pack_jsc_metadata(tensor),
            anchors.tobytes(),
            *packed_streams,
        ]
    )


def unpack_nvq(blob: bytes | memoryview) -> NvqTensor | NvqJscTensor:
    """Deserialize a blob produced by :func:`pack_nvq`."""

    (
        magic,
        encoded_codebook_id,
        sub_bits,
        groupsize,
        axis,
        neuron_len,
        ndim,
    ) = _HEADER.unpack_from(blob)
    if magic not in {_MAGIC, _LEGACY_MAGIC}:
        raise ValueError(f"invalid NVQ magic: {magic!r}")
    sign_mode = "index_parity" if encoded_codebook_id & _INDEX_PARITY_FLAG else "even"
    has_custom_codebook = bool(encoded_codebook_id & _CUSTOM_CODEBOOK_FLAG)
    is_jsc = bool(encoded_codebook_id & _JSC_FLAG)
    codebook_id = encoded_codebook_id & ~(
        _INDEX_PARITY_FLAG | _CUSTOM_CODEBOOK_FLAG | _JSC_FLAG
    )
    if codebook_id not in _ID_CODEBOOK:
        raise ValueError(f"unknown NVQ codebook id: {codebook_id}")
    off = _HEADER.size
    shape = struct.unpack_from(f"<{ndim}q", blob, off)
    off += 8 * ndim
    out = struct.unpack_from("<I", blob, off)[0]
    off += 4

    if is_jsc:
        if has_custom_codebook or sign_mode != "even":
            raise ValueError("invalid NVQ-JSC profile flags")
        if groupsize != 24 or sub_bits != 4:
            raise ValueError("NVQ-JSC v1 requires gs24 and a 4-bit state stream")
        spec = NvqSpec(
            _ID_CODEBOOK[codebook_id],
            groupsize=groupsize,
            sub_bits=sub_bits,
            sign_mode=sign_mode,
        )
        (
            scale_lut,
            bank_for_state,
            codebooks,
            consumed,
            storage_layout,
        ) = _unpack_jsc_metadata_profile(
            memoryview(blob)[off:],
            vector_size=spec.vector_size,
            codebook_entries=spec.codebook_entries,
        )
        if storage_layout == _JSC_GROUP64_LAYOUT_NAME and spec.codebook != "e8_4096":
            raise ValueError("NVQ-JSC group64 storage requires NVQ2J-XL")
        off += consumed
        ng = math.ceil(neuron_len / groupsize)
        nvec = math.ceil(neuron_len / spec.vector_size)
        nsign = math.ceil(neuron_len / 8)
        if off + out * 2 > len(blob):
            raise ValueError("truncated NVQ-JSC neuron anchors")
        neuron_scale = _nonnegative_f16_anchors(np.frombuffer(
            blob, dtype=np.float16, count=out, offset=off
        ), "NVQ-JSC").astype(np.float32)
        off += out * 2
        if storage_layout == _JSC_GROUP64_LAYOUT_NAME:
            state, indices, signs, off = _unpack_jsc_group64(
                blob,
                off,
                out=out,
                groups=ng,
                vectors=nvec,
            )
            state = state.reshape(-1)
            indices = indices.reshape(-1)
            signs = signs.reshape(-1)
        else:
            state, off = _unpack_bits(blob, off, out * ng, 4)
            indices, off = _unpack_bits(blob, off, out * nvec, spec.index_bits)
            signs, off = _unpack_bits(blob, off, out * nsign, 7)
        if off != len(blob):
            raise ValueError(f"invalid NVQ-JSC blob tail: consumed={off}, size={len(blob)}")
        return NvqJscTensor(
            shape=tuple(shape),
            axis=axis,
            neuron_len=neuron_len,
            neuron_scale=neuron_scale,
            scale_lut=scale_lut,
            bank_for_state=bank_for_state,
            state=state.reshape(out, ng),
            indices=indices.reshape(out, nvec),
            signs=signs.reshape(out, nsign),
            codebooks=codebooks,
            base_spec=spec,
            storage_layout=storage_layout,
        )

    spec = NvqSpec(
        _ID_CODEBOOK[codebook_id],
        groupsize=groupsize,
        sub_bits=sub_bits,
        sign_mode=sign_mode,
    )
    codebook = None
    if has_custom_codebook:
        codebook_bytes = spec.codebook_entries * 2
        if off + codebook_bytes > len(blob):
            raise ValueError("truncated NVQ custom codebook")
        codebook = unpack_codebook(spec, blob[off : off + codebook_bytes])
        off += codebook_bytes
    ng = math.ceil(neuron_len / groupsize)
    nvec = math.ceil(neuron_len / spec.vector_size)
    nsign = math.ceil(neuron_len / 8)
    if off + out * 2 > len(blob):
        raise ValueError("truncated NVQ neuron anchors")
    neuron_scale = _nonnegative_f16_anchors(
        np.frombuffer(blob, dtype=np.float16, count=out, offset=off), "NVQ"
    ).astype(np.float32)
    off += out * 2
    sub_scale, off = _unpack_bits(blob, off, out * ng, sub_bits)
    indices, off = _unpack_bits(blob, off, out * nvec, spec.index_bits)
    signs, off = _unpack_bits(blob, off, out * nsign, 7)
    if off != len(blob):
        raise ValueError(f"invalid NVQ blob tail: consumed={off}, size={len(blob)}")

    return NvqTensor(
        spec=spec,
        shape=tuple(shape),
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale,
        sub_scale=sub_scale.reshape(out, ng),
        indices=indices.reshape(out, nvec),
        signs=signs.reshape(out, nsign),
        codebook=codebook,
    )
