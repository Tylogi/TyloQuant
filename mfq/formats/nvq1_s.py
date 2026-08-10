"""NVQ1-S: a 1.34-bit neuron-anchored ternary VQ format.

Every complete group of 24 weights uses exactly 32 logical bits:

    bits  0.. 8  code index 0
    bits  9..17  code index 1
    bits 18..26  code index 2
    bits 27..30  integer sub-scale
    bit      31  codebook bank / delta sign

The four streams are packed across the tensor, so K only needs to be divisible
by eight and no stored K padding is required.
"""

from __future__ import annotations

import base64
import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

from mfq.formats.nvq1_l import IQ1S_TERNARY_2048
from mfq.formats.nvq1_l import _pack_bits as _pack_stream_bits
from mfq.formats.nvq1_l import _unpack_bits as _unpack_stream_bits


def validate_nvq1_s_codebook(codebook: np.ndarray) -> np.ndarray:
    value = np.asarray(codebook)
    if value.shape != (512, 8):
        raise ValueError(f"NVQ1-S codebook must have shape (512, 8), got {value.shape}")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or not np.isin(rounded, (-1, 0, 1)).all():
        raise ValueError("NVQ1-S codebook entries must be in {-1, 0, 1}")
    result = np.ascontiguousarray(rounded, dtype=np.int8)
    if np.unique(result, axis=0).shape[0] != 512:
        raise ValueError("NVQ1-S codebook entries must be unique")
    return result


def validate_nvq1_s_banked_codebook(codebook: np.ndarray) -> np.ndarray:
    value = np.asarray(codebook)
    if value.shape != (2, 512, 8):
        raise ValueError(
            f"banked NVQ1-S codebook must have shape (2, 512, 8), got {value.shape}"
        )
    return np.stack(
        [validate_nvq1_s_codebook(value[0]), validate_nvq1_s_codebook(value[1])],
        axis=0,
    )


NVQ1_S_BOOTSTRAP_512 = validate_nvq1_s_codebook(IQ1S_TERNARY_2048[::4])
NVQ1_S_BOOTSTRAP_512.setflags(write=False)
NVQ1_S_BOOTSTRAP_BANKS = np.stack(
    [NVQ1_S_BOOTSTRAP_512, NVQ1_S_BOOTSTRAP_512],
    axis=0,
)
NVQ1_S_BOOTSTRAP_BANKS.setflags(write=False)


def pack_nvq1_s_codebook(codebook: np.ndarray) -> bytes:
    """Pack one 512-entry ternary table into 1024 bytes."""

    value = validate_nvq1_s_codebook(codebook)
    digits = value.astype(np.uint16) + 1
    shifts = (2 * np.arange(8, dtype=np.uint16))[None, :]
    packed = np.bitwise_or.reduce(digits << shifts, axis=1).astype("<u2")
    return packed.tobytes()


def unpack_nvq1_s_codebook(payload: bytes) -> np.ndarray:
    if len(payload) != 1024:
        raise ValueError(f"packed NVQ1-S codebook must be 1024 bytes, got {len(payload)}")
    packed = np.frombuffer(payload, dtype="<u2")
    shifts = (2 * np.arange(8, dtype=np.uint16))[None, :]
    digits = (packed[:, None] >> shifts) & 3
    if np.any(digits > 2):
        raise ValueError("packed NVQ1-S codebook contains a non-ternary digit")
    return validate_nvq1_s_codebook(digits.astype(np.int8) - 1)


def pack_nvq1_s_banked_codebook(codebook: np.ndarray) -> bytes:
    value = validate_nvq1_s_banked_codebook(codebook)
    return pack_nvq1_s_codebook(value[0]) + pack_nvq1_s_codebook(value[1])


def unpack_nvq1_s_banked_codebook(payload: bytes) -> np.ndarray:
    if len(payload) != 2048:
        raise ValueError(f"banked NVQ1-S codebook must be 2048 bytes, got {len(payload)}")
    return validate_nvq1_s_banked_codebook(
        np.stack(
            [unpack_nvq1_s_codebook(payload[:1024]), unpack_nvq1_s_codebook(payload[1024:])],
            axis=0,
        )
    )


# Provisional universal table trained on a deterministic 64x1024 Gaussian
# matrix (seed 20260717), with a disjoint 16x1024 validation matrix.
_NVQ1_S_SYNTHETIC_BANKS_B85 = """
1S*k+3L0gl5NQ<%8U>MrNnw~mm}pT_X(ploW)%dX5rJ80K?)jX6hT5&7ExwJAxR-smIYuWRhbZ#WEKz@5L6k2P$fW!fQ3axDHQ=E
Pz4l_R23GHl2`?iRwMu<8f6w>1f`ZxNhA_!R8$qHRY4^ISYcEF7(_@xS%^Ye5s?5vWCa#v6hu%G0HIMz27+NmP(ekcg+&o2h*W76
SSAn=RZ3QvWg%D=LPbP|sudX!WtkM1AVgMZK^U1;QCS9LT9H;|hE-JoMp0D)m4FFVS!Q4%qy?ZtCI(rC0g@q=WF~=BNfu;*BoGiH
5Ls$M6%v(NkpxmeCPHCIpb=DLNkCaZL{cP45><sFAtj|D1wa)^Wm#A$QD8}xWCVd`83dqJK!BBn5rruxQC209R%BE`P*f3CL{tG5
6-bqpmRVvMAQeR*RcMrH1_i2EVN#YsmQh7k22=zVl?6Znl@(c1q*Y~P0fkYJRbWJEP-Re+AyJtTf<+YonnqAnnQ0V8WK>j<lt2*y
R)7SNRt1SsWFaLP1*N5?Xjzd}DHLQ<VM&-$P=;Csg_V?25vfsuRRobCMI;1NMFdr4Nkv5&L|K(tU;$MUP*#<Z6d_<$6roy>R#qjH
K$%8HQ5F@cp+r@rgir=jNL2<=l@L}*VFgi^77<Yi5kO^BSr$o!36)U=kx@~UniP};T2NU~NSTskW)u}hU>OiqiD_s>1qBop6ciRD
StJr<R%Vd`X(1tGl^79GRUt-J1yqGnQDspgiB%d>kYxl~1z|;0MOlPagh7N+RV9gGR)ra66=Y~gWR+Ew1yO-zL|}m#Vo_E=C6Yl^
1p!)B5oJL|L>3lRC00@?B~VosRTWtgRY*}(NR~laVFnQqlvSc6B$Om+K~@z~VN_;;P!UFwmIM~42vud2Q4tkXR2C&gBxIEo6%-Uv
QDG1!XcbUJkO5RtNfJ~LWf=xmg_c2N6^Uh40u@ymWtBlHMo?uEMP!khB?U!gWmbrll~hzjq#2<BWl>dFWkD5CMU+`ZRhmR<MMOy^
6l7%-RH;^3Rb~NUgjH5iWo4liW@KR%RYnCCSXEVlRYfHgRTu=6RRxt+g<4^iLQxQzT2d(nL0Fbl5MW`65l|@@ML-rtRZv-3Rz(z6
rD267RS^YQRb-V2K`B*4g_r=9QCN{wK~!XsP!$zLWkv-Rg;fz%RasF)R+W*JSX7lzNKs`K78ZpSQ58jnKvh+SMI~5JQBhSCWKoqt
RX}ByR7F)*6+~85K@~+6kQEhGR8~b*Rgsld1XUGL6h%Z~MNmc%MFm7vNmW!7Sy5C~1yw{9QB_4zRaI3{C2EQmVp<_!QCK0STAC$>
q^W9(sA*ZDl@&pN6quSxC|QM78kGc8X&GvzWCa$GWR{9lN+hZordnEAW@3q1MFIp$S_NsTB%(=)phje(R3K4RMks`$i4`fTsF+eD
SV|?8VWf&-mH|aomX%phWmr*WYE`L~Ns?9>VP#fDMJOUtQCWtHR3(y;hFOJ0Dyl+MWLX+gLMWw}5~YcmR9KW1kZEN>l_rK(CZ(a8
MU<AHs+N`%f|`bE6_#WeT1ZGnl^T&rg{7I6rl_TnQ7VB&Q6gw%R8(11Wug{Y8C9WFY9xh3rA1Ozlo%SSVn$j~lonQoiB?6WL}3=G
lv-vfVOk-jsZf+rp;eWcltgHjra~o@ScI9Tq)B2~g_V?8Vp6DqT7ptpMir`NP-3EK5r&8gRaO#}Qdklig<7Oa1fUt1U`1AyXjNrY
R9OiTg&L)4XjvA4NmXDeSdx-rW>#q^Q9=QsSd~Rk7E)MdR;X!IWL8+25L%`gMrnm+0%cU7MI=_L0g{-QQUxhhR7EA3rd4K1Sw(4p
5tdn&ML<PaAZ1x$gjre@Pytj#p{c4CfJ7x_S&>x{kr0_wLPk|sl$cRkp+puW8Wou$SW=d06;WzY8B__CWGIyum6lbS7D-xVg;AL%
QkGR2g_2QL5kXX%k%nq%2Bk`BkyS}5m4#NMVo6mcP*!GTX=GUzS!Pm@npP1R5=J7LMTKca6%wTqDkwlyN@YY<TA^lCm1KnlQD_#H
mRgWtVU<`>Mn*|e8dVikL0OVo1(iXSQAK1WriErzm7!%?s*#!r1XW2UWtCY{kye$ImZD@;Noi?FQBpw_re#)Uk(H`dWl)i1RYg%5
6;(x28BtkfR;HFks)ngiRYGc6Rz@k6X%>P-Rtcphl~^jG1W{FqkX2$8mRP11RD~53fN5oxQkfconx!OaDwbBI2`XAuVQE$v31O8<
R%DS`mRS{PVwOrsk)|1Gs8vx0MOKtjSy&ilM5YC46j3H+RzYPIrb$X!RuNQHMx_Z^lz~>6Bv?|a5oTr?nOT}e6@?O%R!WpvrB-E$
Sz1O_W|WjAs!<UsWtvq<Qc0BsQDvo3QDGSs6<L|3Rgn={rBoGKm1RXmY8h1qgj!{xDORXhWm#rsm}aG<RazBgl~#$CWtmz^RFzm+
mQ_YsT2!Vgl^IoOYFb69W|>qKmKkMbRaBK#6&Y0(MP*f%Rzg-~k!4w>RgzX(nPo&4MI~iXX_aMFRuz>+T4|+~Wu+OFSw@<bR#v4|
Rh5}mWol(+QEFA8Srt;1m6e%Q7FCuNSy@(5X;oEGrKMI?Sye@5n5AZwRb^FG
"""
_NVQ1_S_SYNTHETIC_BANKS_SHA256 = "beb341195ce650979d2640b6a494652047d8a371ea0af66183fc5c9f38a5a37e"


def _decode_synthetic_banks() -> np.ndarray:
    payload = base64.b85decode(
        "".join(_NVQ1_S_SYNTHETIC_BANKS_B85.split()).encode("ascii")
    )
    if hashlib.sha256(payload).hexdigest() != _NVQ1_S_SYNTHETIC_BANKS_SHA256:
        raise RuntimeError("NVQ1-S synthetic codebook checksum mismatch")
    table = unpack_nvq1_s_banked_codebook(payload)
    table.setflags(write=False)
    return table


NVQ1_S_SYNTHETIC_BANKS = _decode_synthetic_banks()
NVQ1_S_TABLE_BYTES = 2048


@dataclass(frozen=True)
class Nvq1SSpec:
    groupsize: int = 24
    sub_bits: int = 4
    delta: float = 0.15625

    def __post_init__(self) -> None:
        if self.groupsize != 24:
            raise ValueError("NVQ1-S groupsize is fixed at 24")
        if self.sub_bits != 4:
            raise ValueError("NVQ1-S sub-scale is fixed at 4 bits")
        if self.delta != 0.15625:
            raise ValueError("NVQ1-S delta is fixed at 5/32")

    @property
    def vector_size(self) -> int:
        return 8

    @property
    def index_bits(self) -> int:
        return 9

    @property
    def label(self) -> str:
        return "NVQ1-S"

    def stream_nbytes(self, out: int, neuron_len: int) -> int:
        if out <= 0 or neuron_len <= 0 or neuron_len % 8:
            raise ValueError("NVQ1-S dimensions must be positive and K divisible by 8")
        ng = math.ceil(neuron_len / self.groupsize)
        nvec = neuron_len // self.vector_size
        anchors = 2 * out
        scales = (out * ng * self.sub_bits + 7) // 8
        indices = (out * nvec * self.index_bits + 7) // 8
        deltas = (out * ng + 7) // 8
        return anchors + scales + indices + deltas

    def payload_nbytes(
        self,
        out: int,
        neuron_len: int,
        *,
        include_codebook: bool = True,
    ) -> int:
        codebook = NVQ1_S_TABLE_BYTES if include_codebook else 0
        return codebook + self.stream_nbytes(out, neuron_len)

    def bpw(
        self,
        neuron_len: int,
        *,
        out: int = 1,
        include_codebook: bool = True,
    ) -> float:
        return (
            8.0
            * self.payload_nbytes(out, neuron_len, include_codebook=include_codebook)
            / (out * neuron_len)
        )


NVQ1_S = Nvq1SSpec()


@dataclass
class Nvq1STensor:
    spec: Nvq1SSpec
    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    neuron_scale: np.ndarray
    sub_scale: np.ndarray
    indices: np.ndarray
    delta_sign: np.ndarray
    codebook: np.ndarray | None = None

    @property
    def payload_nbytes(self) -> int:
        return self.spec.payload_nbytes(self.neuron_scale.size, self.neuron_len)

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


_MAGIC = b"NQ1S"
_VERSION = 1
_HEADER = struct.Struct("<4sBBHiiI")


def _validate_tensor(tensor: Nvq1STensor) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    if not shape or not 0 <= tensor.axis < len(shape):
        raise ValueError(f"invalid NVQ1-S shape/axis: {shape}, axis={tensor.axis}")
    out = int(np.asarray(tensor.neuron_scale).size)
    if int(np.prod(shape)) != out * tensor.neuron_len:
        raise ValueError("NVQ1-S shape does not match neuron dimensions")
    if tensor.neuron_len % 8:
        raise ValueError("NVQ1-S neuron length must be divisible by 8")
    ng = math.ceil(tensor.neuron_len / 24)
    nvec = tensor.neuron_len // 8
    with np.errstate(over="ignore", invalid="ignore"):
        anchors = np.asarray(tensor.neuron_scale, dtype=np.float32).astype(np.float16)
    if not np.isfinite(anchors).all() or np.signbit(anchors).any():
        raise ValueError("NVQ1-S neuron anchors must be finite and non-negative in FP16")
    expected = {
        "sub_scale": (out, ng),
        "indices": (out, nvec),
        "delta_sign": (out, ng),
    }
    for name, expected_shape in expected.items():
        actual = np.asarray(getattr(tensor, name))
        if actual.shape != expected_shape:
            raise ValueError(f"bad {name} shape: {actual.shape}, expected {expected_shape}")
    if np.any(np.asarray(tensor.sub_scale) >= 16):
        raise ValueError("NVQ1-S sub_scale exceeds four bits")
    if np.any(np.asarray(tensor.indices) >= 512):
        raise ValueError("NVQ1-S codebook index exceeds nine bits")
    if np.any(np.asarray(tensor.delta_sign) > 1):
        raise ValueError("NVQ1-S delta_sign must contain only 0 or 1")
    validate_nvq1_s_banked_codebook(
        NVQ1_S_SYNTHETIC_BANKS if tensor.codebook is None else tensor.codebook
    )
    return out, ng, nvec


def pack_nvq1_s(tensor: Nvq1STensor) -> bytes:
    """Serialize self-contained NVQ1-S streams without K padding."""

    out, _, _ = _validate_tensor(tensor)
    shape = tuple(int(value) for value in tensor.shape)
    codebook = NVQ1_S_SYNTHETIC_BANKS if tensor.codebook is None else tensor.codebook
    return b"".join(
        [
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                tensor.spec.sub_bits,
                tensor.spec.groupsize,
                tensor.axis,
                tensor.neuron_len,
                len(shape),
            ),
            struct.pack(f"<{len(shape)}q", *shape),
            struct.pack("<I", out),
            pack_nvq1_s_banked_codebook(codebook),
            np.ascontiguousarray(tensor.neuron_scale, dtype="<f2").tobytes(),
            _pack_stream_bits(tensor.sub_scale, tensor.spec.sub_bits),
            _pack_stream_bits(tensor.indices, tensor.spec.index_bits),
            _pack_stream_bits(tensor.delta_sign, 1),
        ]
    )


def unpack_nvq1_s(blob: bytes | memoryview) -> Nvq1STensor:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated NVQ1-S header")
    magic, version, sub_bits, groupsize, axis, neuron_len, ndim = _HEADER.unpack_from(blob)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("invalid or unsupported NVQ1-S header")
    if sub_bits != NVQ1_S.sub_bits or groupsize != NVQ1_S.groupsize:
        raise ValueError("unsupported NVQ1-S stream profile")
    if ndim <= 0 or neuron_len <= 0 or neuron_len % NVQ1_S.vector_size:
        raise ValueError("invalid NVQ1-S dimensions")

    off = _HEADER.size
    shape_bytes = 8 * ndim
    if off + shape_bytes + 4 > len(blob):
        raise ValueError("truncated NVQ1-S shape")
    shape = tuple(struct.unpack_from(f"<{ndim}q", blob, off))
    off += shape_bytes
    out = struct.unpack_from("<I", blob, off)[0]
    off += 4

    if off + NVQ1_S_TABLE_BYTES > len(blob):
        raise ValueError("truncated NVQ1-S codebook")
    codebook = unpack_nvq1_s_banked_codebook(bytes(blob[off : off + NVQ1_S_TABLE_BYTES]))
    off += NVQ1_S_TABLE_BYTES
    anchor_bytes = 2 * out
    if off + anchor_bytes > len(blob):
        raise ValueError("truncated NVQ1-S neuron anchors")
    neuron_scale = np.frombuffer(blob, dtype="<f2", count=out, offset=off).astype(np.float32)
    off += anchor_bytes
    ng = math.ceil(neuron_len / NVQ1_S.groupsize)
    nvec = neuron_len // NVQ1_S.vector_size
    sub_scale, off = _unpack_stream_bits(blob, off, out * ng, NVQ1_S.sub_bits)
    indices, off = _unpack_stream_bits(blob, off, out * nvec, NVQ1_S.index_bits)
    delta_sign, off = _unpack_stream_bits(blob, off, out * ng, 1)
    if off != len(blob):
        raise ValueError(f"invalid NVQ1-S blob tail: consumed={off}, size={len(blob)}")
    tensor = Nvq1STensor(
        spec=NVQ1_S,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale,
        sub_scale=sub_scale.reshape(out, ng),
        indices=indices.reshape(out, nvec).astype(np.uint16),
        delta_sign=delta_sign.reshape(out, ng),
        codebook=codebook,
    )
    _validate_tensor(tensor)
    return tensor
