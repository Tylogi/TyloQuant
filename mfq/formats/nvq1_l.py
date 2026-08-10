"""NVQ1-L: neuron-anchored 8-D ternary lattice quantization.

The codebook is llama.cpp's exact 2048-entry IQ1_S ternary grid. NVQ1-L changes
the scale hierarchy: one FP16 anchor per output neuron and one integer relative
scale plus one delta sign per small group.

This module defines the packed disk representation only. It does not provide a
CUDA kernel or make runtime-throughput claims.
"""

from __future__ import annotations

import base64
import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

_IQ1S_GRID_B85 = """
000621po*D3IGuR6#yUrA^<1=DgZ?QQ2<l`RsdxHfB=F3hyaQJl>ndsq5!A>ssIE51pyHO6af|i837srB>_PJMgd6yQUO%~S^;4J
WC3OYXaR)*kpYwemI0*!00II62m%TM6#^gvA_6D^Dgs3UQ36>4WCCddfC7R7hyscNkph(hnF62!q5`M_ssa%N6a*FoB?LhPNd#2{
S_EVSWdxA~nFOT-0R;pF1qB8L6$Kat8U-l@Km|nwN(E2_Q3X^5RRvZBSp{GPVg+RdXa#Bofd!ETl?9jungyW+qy?n~rUj`56b2av
K?XzyPzF*4RR&lFU<P3ZW(H{ng$9uZlm?jw00;sI2nY%Y6$l^*A_yo5DhNdgQ3zHDWe9)>f(VERiU^empa`M}s0gYE1ql%e6bTs#
Bnc%6K?zU^Q3+KEVF_djX$g@DlnIsznF*x{015&M2nq@c6$&5<A_^!9DhfpkQ3_cKVG3mmfC_>Ng$jrYiVBqqpbDZ2s0ykI5D*a%
6c8B@BoHMKK@da`P!LrRSP)?lWDsQ#X%LYRln|B>p%A4I0TBcd1`!Dn5D^j)6%iN_ArT}IDG@~xN)b>IQ4vxRR1sAXRuNedU=d{z
gb|Vvl@Xy4q!AGk6cQE^B@#dqMiNO9QW8}XSQ1(iWD;f)g%XhxloFN_r4j)Y1{4Vt6ciN{7!(;4ArvMQK@>$4MifXCN)%BPR1{Sd
RuowjViaW*XcUALi4>3&loXW|m=u{5niQcFq!gtTsT2Yg1r-Vv5fu~_6%`g085JNEA{8YSC>1IdK@~(5MHNOBQ58}ZR25YfRux$l
S`}dxWEEu<W)*1_fE9ujgcXGqh!u(zkQI>?logc~mKB*5niZfGq7|hT0Tu)n1r`Pt6&4m27#129Ar>YUKo&w4L>5IBNES*KQ5IDe
Ru)(mSr%ayWEN!>XclP}Y8HkTiWZU<l@_EHsTL6!78oTMK^R0BMi@yLP#9GhSQuIuU>IQ-WEf@`X&8kVkr<R1r5F$y5*Za48W|xO
B^g2)L>WaHNEuNXR2fwnRvB3tS{Yy&Wf^K2i5Za@k{Oj5m>HoNrWvUj2^tm}BpN0fL>fjKNg7ZZQW{knSQ=p(W*TW4g&L6>mKvHG
03ZS&2p|d|6(ArWA|NFoC?F~zMIcciSs-B`Wgvhcf*^<>iXfFBpdg|kr68yvsvrd+5g`;I86hPhLLo#UNg-7sSRq;=VIgE8Wg%uE
g&~k3mLZuTr6B<#2qFq15h4{LAR;0nC?YB%MIuomRw7v<Wg>whh$4z2ks_5Mpdz9os3NK&1tb(C79<%YB_u>7MI=TfNhDGvRU}v>
S|ny9g(Q(Alq8uXp(Ldw2_+RJAtfm#KqW;bNF`AvR3%j<StVj6WhH1Oi6xLFlqHoVm?fGep(UgxrX{H*1tt(C5+)fYB_=^8NhVb$
U?yQEX(ogwh9;0EnkE1!0w@S53MdsQASfazC@3l@MJQ1yR48RAfGC0}h$xCEpeUj!s3@u^2`LdN6e$@gB`HQJNhwk(RViU9WGQAT
X(@#%kSUfanJJ_xr6~X^0xAe93MwEfA}S~<Dk?=PQ7Tp{St?~JfGUD2h$@OIl`5brqAI8=swxFQ5kM9|B|u3)Qb1KeSU_4pVL)U+
W<ZocnLwdHra%Be0YL;o20;ly5<wL~7C{(68bKjJCP67dML|eGN<mRUR6$iiRzX=oT0vz&XhCX0fkA{phCz@)l0lV0ph2NQqCo{h
5keF~7D6RLK|)eORYF=qWI|~|i9(b@r9uHj6+{_CDMUp?NJL6RQAAWkRYX=qVMJm?WkhI1YD9rVhD3=(kVKM1l|-OKp+u=f0Yw5u
1w{%)5k(Y56-5?B8ATvPB}FPlK}AGGMMXwMNkvdaQAJcmRYg`sSVdVyVMSy`WkqI1X+?rXghho#h((b_ltq<AmPMIGnnk5Ws70zp
0Y(Kz2}Tq~6-F3F8b%>TBt|JlKt@7FMMg+QP)1QkQbtuqRz_JyVn$^~XhwlXg+_`-ltz_Cp+=-erbY!w5l9tC8b~EbLP$wSP)Jos
SV&<=WJqR6X-I`gkw}zCmPnaMrAPru1xW@;3P})26iF3H7)c>XBuOSoKuJYON=Z>kQb|-vRY_J!SxI0?Vo7C5W=U#EhDnJ@l1Y_G
mPwdNp-H4krb($u7D_})MoLLaRZ3V&T1sR}X-bqzrAh%%1W*M~22cr55>ORT8c-onBv2_(Ku|?cNKjEwR8Un=R!~_`Wl(5PhER!6
l~9;apirSurckL+1yKl52~i4B5m6LT6;T#Z7*QEfAW<bzCQ&F+Dp5gEL{UXiMo~#oP*G7)Qc+Y<RZ&(^SW#I~T2Wz9WKm^NW>INT
f>DK0kx`UUl~I;anNgrorBSL;0a6B15>gdX8d4!rBvL|BMN&#qQBqV=RZ><`SyEzBWm1JwkW!LTl~S2fno^`v1XKl75mXdZ6;u{f
7*rWlAyg$)C{!v`K~zLkMN~#qNmNQyP*hP=R8&<|R#aG2SyWn7VN_yNWK?BTW>jfZfK-T7id2zQlvI^emQ<Nkp;V+)rBtd^0aXH3
1XTr922}}F5LFRX5>*sc6;&2h8C4opAyp(*B~>O>DOEsKK~+LkL{&vqMpZ~vNmWo)QB_h^R8>_~R#jM4Syft9U{zsNWL0HVW>sia
X;o@ffmMW6g;j}FidB$RkyVscl~tBim{plop;e?+rB$X?sZ{`00agN51Xc!C2v!MJ5mppd6;>El8CD=xAyy(*BvvI>CRQj`DON#N
MOH>uNLEQ!N>)%-QC3n`R9011R#sS6SyozBVOC^TWmaicf>woAhE|AHiB^hMkye#fnpUD#rB<d^s8*>~1Xu-F23QGL5Lgvh7+4`#
Dp*BWNLWf(QCL)1RajP7SXfzDU|3>UWLRZbf>?=IkXVvfl~|Zqp;)O{0a*fB1z8GN5m^*j6<HQp8CfM+L0LpuMOj8!Nm)=?QCU)1
R9RJ7R#{kCSy@_HVOeBZWm#rfX<30;g;|MNky(^km06Zqm|2-wrC9`82wD|d8d@M)C0Zt0DOyEZNLoo+QCd}6R$5qESz2ORWm;%j
YFdF>idvFdm0Ffsm|B`zp;`rC6krx$8DJ$~L|{;0RbW<NSYTRUVPIrnW?*Sxfnb(krC<SJ1YrhY31Jdp6=5M^B4H+BDPcunNnuf8
Rbf_PSz%>iW?^b!gkg$dl3|r$p<$+BsbLXf7GfD<Kw?2+Mq*WBR$^FUU}A-0kz$r&rD73e5@Z!*7GxS^A!H_GDP%xoLS#i`NMuT6
QDjtPRb*CVS!7ydU}R!sWn_e8g=C3jkYti%lw_4;mSmV@nq;A5q-3dN1!V|j3S|*x6=fD>8D&IeMP){1No7!FQDszRRb^IXS!H2m
WMyS#W@Tw*hGmImie-^ym1UM?nPr+~qGhFJre&yQ0%i$j6=oo2CT1vRDP}-sMP^85N@h`JRAyCXR%TdcT4rEoWoBq*fM$Ybg=UIo
lxCJ@m}Z$~pk}0Ire>;U7HAo0C1^otQfO6ZT4-TtX=sILk!YA`rf30a1Zfay6=@-9BxxpTDQQ4yL1{&4MrlZCQE60ZRcTgfS!rNt
Woc?@f@y?lifNT;p=qRPrD>^Y5o#7{7-~UkL~2QDP-<0bSZZ2oWNKw<X=;XQlxmo2nrfzM0DuC32!INJAb=u(D1a(}MSxI%QGir{
R)AT6Wq^Qyf`EvCihz}Xpn#%)sDP?~1%VNP6oD3j8G$8%L4ibpNr6y-Qh`;0R)JW7S%GAMW`Sveg@K8Il!2CknSrH&0D=O72!aZN
6@nmwB7!J_DuPjhRDxN8WrBc$f`W*Gih`Abpn{@;sDi436oeUsL4-tvQG`{5T7+SQWQ1vil!Tdt0fh;L5``6o8igi>DTP3VL4`$y
NQF^_RE1TAR)txGT7_kWW`$^lYK4J?goTEMiG`4bl7*Fpn1!Z=5r!6q8HOc>L54(zNrp;>P=-~8S%zAMVTNXgYKDb|k%pRvq=o>9
0*DBR3WybmAc!J}D2OVEL5M|&QHWHCS%_tbX^4P`f{2KSiinknpopS~sEDeF1&IcU5s4Iu7KtT^L5W0(Mu|y@P>E8BRf$@OVTojk
g^8AlnTe%|0Ez;N2#N}d6^bB=B8n)ADvCvlQHoTGR*Ha%f{KWWii(wrpo*f3sEVqJ1&|St7LXW_8IUEAL6Am}Nsv{LSddzfX^?7=
g^-bul#rH?nUJNB0g(ie29XJo5Rny;7?B#0A(14JCXp$TK#@g}P?1rQRFPGYR*_keVv%K$gpr1kl982>n3188q>-kPsgVVe5t0;=
8ImQEM3P35Ns>^KQj%4YSdwOvX_AGKl#-T`sgePd1e6Ap5R?^^7?dWIK$J?9QIu4aRg_kgSd>|mV3c8$Vw7c+gp`Jql9ZH`m6Vv2
p_HkV0F?!m2$c$z5S0;?6qOZ~7L^&5A(bVSDU~XfL6t<6MU_UCNtIBQQI%4aRFzegR+U(lS(RFqVU=W+WtC=?X_bMMg_Vevk(H8_
l$Dj0mX(>6nw6lHqLrnUsFkXf0hR=o5SA5|8I~ZHCYC9dMV3gGNtRKTQkGSgR+d?oWtM1`f|i7qiI$3%l9rT~m6oKIrk1Ie1(+6?
8JH!QL6}CEP?%DfRhU+oT9{;*WtfGSk(id4nV6-R1epez37HU?5}6g57?~QGAekYVB$+0eK$$|BMVUyMN|{laRGC$oR+(9uVwq)`
W|?Z4ftiGvkeQO1m6@8Ep_!(c1)3F_C7MK<Mw&^QP?}YmSejv)g_@C?l$w>AmYM*d0-y+>3ZNCBAfO_kD4;5!MW9ijRG?O%S)hQR
f}n_?ilCLCprE3lsGzE#1)&L{5up^J7NHrT8lgs^Nug1pRiRj+TA^W~WTAzjkfD;HmZ6!U0-^|_3Zf9A8KNShD55H&MWRunR-##=
Wukzhf})6`ilUXGprWFpsG_Q(8KfnoL8L^aP^46(Ris#?TBKp5Wu#`LXryVRg`@_h38fIF5~UTT7^NnqDWyWCMWs=tRHap=R;5{`
Wu<DRfu)3{g{6k2iKUXIm8F=a1*Q?E7N#1eA*Lm!M5acjN~Th#Ri;*^Sf*g6Vy1?skfxQUmZq7ep{As!rltU>0;mY63aB8cBB&^+
DyUJYRH#;{S*U=hf~bh7il~*Sps1p#sHm!_1*sIN8L1(uC8<HFP^nd^TB%{FW~phZkg1lb0ICA22&xLIAgUs&D5@%|QL0p`R;qxi
f~ttBimH{WpsJ#(sH&=}
"""
_IQ1S_GRID_SHA256 = "e9ffebfc997ad6023063a667c0bf4eedd8ebf7f56b3da399cbc7d1b1f7ea52c6"


def _decode_iq1s_grid() -> np.ndarray:
    raw = base64.b85decode("".join(_IQ1S_GRID_B85.split()).encode("ascii"))
    if hashlib.sha256(raw).hexdigest() != _IQ1S_GRID_SHA256:
        raise RuntimeError("NVQ1-L codebook checksum mismatch")
    encoded = np.frombuffer(raw, dtype="<u2")
    if encoded.size != 2048:
        raise RuntimeError(f"NVQ1-L codebook has {encoded.size} entries, expected 2048")
    shifts = 2 * np.arange(8, dtype=np.uint16)
    digits = (encoded[:, None] >> shifts[None, :]) & 3
    if np.any(digits > 2):
        raise RuntimeError("NVQ1-L codebook contains a non-ternary digit")
    grid = np.ascontiguousarray(digits, dtype=np.int8) - np.int8(1)
    grid.setflags(write=False)
    return grid


IQ1S_TERNARY_2048 = _decode_iq1s_grid()


def validate_ternary_codebook(codebook: np.ndarray) -> np.ndarray:
    value = np.asarray(codebook)
    if value.shape != (2048, 8):
        raise ValueError(f"NVQ1-L codebook has shape {value.shape}, expected (2048, 8)")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or not np.isin(rounded, (-1, 0, 1)).all():
        raise ValueError("NVQ1-L codebook entries must be in {-1, 0, 1}")
    result = np.ascontiguousarray(rounded, dtype=np.int8)
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("NVQ1-L codebook entries must be unique")
    return result


def pack_ternary_codebook(codebook: np.ndarray) -> bytes:
    value = validate_ternary_codebook(codebook)
    digits = (value.astype(np.int16) + 1).astype(np.uint16)
    shifts = (2 * np.arange(8, dtype=np.uint16))[None, :]
    packed = np.bitwise_or.reduce(digits << shifts, axis=1).astype("<u2")
    return packed.tobytes()


def unpack_ternary_codebook(payload: bytes) -> np.ndarray:
    if len(payload) != 4096:
        raise ValueError(f"packed NVQ1-L codebook must be 4096 bytes, got {len(payload)}")
    encoded = np.frombuffer(payload, dtype="<u2")
    shifts = (2 * np.arange(8, dtype=np.uint16))[None, :]
    digits = (encoded[:, None] >> shifts) & 3
    if np.any(digits > 2):
        raise ValueError("packed NVQ1-L codebook contains a non-ternary digit")
    return validate_ternary_codebook(digits.astype(np.int8) - 1)


@dataclass(frozen=True)
class Nvq1LSpec:
    """Storage parameters for one NVQ1-L tensor."""

    groupsize: int = 24
    sub_bits: int = 3
    delta: float = 0.125

    def __post_init__(self) -> None:
        if self.groupsize <= 0 or self.groupsize % 8:
            raise ValueError("NVQ1-L groupsize must be a positive multiple of 8")
        if not 1 <= self.sub_bits <= 8:
            raise ValueError("NVQ1-L sub_bits must be in [1, 8]")
        if self.delta != 0.125:
            raise ValueError("NVQ1-L currently supports the IQ1_S delta 0.125 only")

    @property
    def vector_size(self) -> int:
        return 8

    @property
    def index_bits(self) -> int:
        return 11

    @property
    def label(self) -> str:
        return f"NVQ1-L-T8-S{self.sub_bits}"

    def payload_nbytes(self, out: int, neuron_len: int) -> int:
        """Packed bytes excluding the self-describing blob header."""

        if out <= 0 or neuron_len <= 0:
            raise ValueError("NVQ1-L tensor dimensions must be positive")
        ng = math.ceil(neuron_len / self.groupsize)
        nvec = math.ceil(neuron_len / self.vector_size)
        anchors = out * 2
        scales = (out * ng * self.sub_bits + 7) // 8
        indices = (out * nvec * self.index_bits + 7) // 8
        deltas = (out * ng + 7) // 8
        return anchors + scales + indices + deltas

    def bpw(self, neuron_len: int, *, out: int = 1) -> float:
        return 8.0 * self.payload_nbytes(out, neuron_len) / (out * neuron_len)


NVQ1_L_T8_S3 = Nvq1LSpec(groupsize=24, sub_bits=3)
NVQ1_L_T8_S4 = Nvq1LSpec(groupsize=24, sub_bits=4)


@dataclass
class Nvq1LTensor:
    """Packed-domain NVQ1-L tensor before byte serialization."""

    spec: Nvq1LSpec
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
        custom = 4096 if self.codebook is not None else 0
        return custom + self.spec.payload_nbytes(self.neuron_scale.size, self.neuron_len)

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    if not 1 <= bits <= 16:
        raise ValueError(f"unsupported packed width: {bits}")
    values_u16 = np.ascontiguousarray(values, dtype=np.uint16).reshape(-1)
    if values_u16.size and np.any(values_u16 >= (1 << bits)):
        raise ValueError(f"value does not fit in {bits} bits")
    shifts = np.arange(bits, dtype=np.uint16)
    rows = ((values_u16[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(rows.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(blob: bytes, off: int, count: int, bits: int) -> tuple[np.ndarray, int]:
    if not 1 <= bits <= 16:
        raise ValueError(f"unsupported packed width: {bits}")
    nbytes = (count * bits + 7) // 8
    end = off + nbytes
    if end > len(blob):
        raise ValueError("truncated NVQ1-L bit stream")
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint32)
    values = (stream.astype(np.uint32) * shifts).sum(axis=1)
    return values.astype(np.uint16), end


_MAGIC = b"NQ1L"
_PROFILE_IQ1S_GRID = 1
_PROFILE_CUSTOM_TERNARY = 2
_HEADER = struct.Struct("<4sBBHiiI")


def pack_nvq1_l(tensor: Nvq1LTensor) -> bytes:
    """Serialize NVQ1-L metadata and all tightly packed streams."""

    spec = tensor.spec
    shape = tuple(int(value) for value in tensor.shape)
    if not shape or not 0 <= tensor.axis < len(shape):
        raise ValueError(f"invalid NVQ1-L shape/axis: {shape}, axis={tensor.axis}")
    out = int(tensor.neuron_scale.size)
    if int(np.prod(shape)) != out * tensor.neuron_len:
        raise ValueError("NVQ1-L shape does not match neuron dimensions")

    ng = math.ceil(tensor.neuron_len / spec.groupsize)
    nvec = math.ceil(tensor.neuron_len / spec.vector_size)
    expected = {
        "sub_scale": (out, ng),
        "indices": (out, nvec),
        "delta_sign": (out, ng),
    }
    for name, expected_shape in expected.items():
        actual = np.asarray(getattr(tensor, name))
        if actual.shape != expected_shape:
            raise ValueError(f"bad {name} shape: {actual.shape}, expected {expected_shape}")
    if np.any(np.asarray(tensor.sub_scale) >= (1 << spec.sub_bits)):
        raise ValueError("NVQ1-L sub_scale exceeds its packed width")
    if np.any(np.asarray(tensor.indices) >= 2048):
        raise ValueError("NVQ1-L codebook index exceeds 11 bits")
    if np.any(np.asarray(tensor.delta_sign) > 1):
        raise ValueError("NVQ1-L delta_sign must contain only 0 or 1")
    with np.errstate(over="ignore", invalid="ignore"):
        anchors = np.ascontiguousarray(tensor.neuron_scale, dtype=np.float16)
    if not np.isfinite(anchors).all() or np.signbit(anchors).any():
        raise ValueError("NVQ1-L neuron anchors must be finite and non-negative in FP16")

    custom_codebook = (
        pack_ternary_codebook(tensor.codebook) if tensor.codebook is not None else b""
    )
    parts = [
        _HEADER.pack(
            _MAGIC,
            _PROFILE_CUSTOM_TERNARY if tensor.codebook is not None else _PROFILE_IQ1S_GRID,
            spec.sub_bits,
            spec.groupsize,
            tensor.axis,
            tensor.neuron_len,
            len(shape),
        ),
        struct.pack(f"<{len(shape)}q", *shape),
        struct.pack("<I", out),
        custom_codebook,
        anchors.tobytes(),
        _pack_bits(tensor.sub_scale, spec.sub_bits),
        _pack_bits(tensor.indices, spec.index_bits),
        _pack_bits(tensor.delta_sign, 1),
    ]
    return b"".join(parts)


def unpack_nvq1_l(blob: bytes) -> Nvq1LTensor:
    """Deserialize a blob produced by pack_nvq1_l."""

    if len(blob) < _HEADER.size:
        raise ValueError("truncated NVQ1-L header")
    magic, profile, sub_bits, groupsize, axis, neuron_len, ndim = _HEADER.unpack_from(blob)
    if magic != _MAGIC:
        raise ValueError(f"invalid NVQ1-L magic: {magic!r}")
    if profile not in {_PROFILE_IQ1S_GRID, _PROFILE_CUSTOM_TERNARY}:
        raise ValueError(f"unsupported NVQ1-L profile: {profile}")
    if ndim <= 0:
        raise ValueError(f"invalid NVQ1-L ndim: {ndim}")

    off = _HEADER.size
    shape_bytes = 8 * ndim
    if off + shape_bytes + 4 > len(blob):
        raise ValueError("truncated NVQ1-L shape")
    shape = tuple(struct.unpack_from(f"<{ndim}q", blob, off))
    off += shape_bytes
    out = struct.unpack_from("<I", blob, off)[0]
    off += 4

    spec = Nvq1LSpec(groupsize=groupsize, sub_bits=sub_bits)
    codebook = None
    if profile == _PROFILE_CUSTOM_TERNARY:
        codebook_bytes = 4096
        if off + codebook_bytes > len(blob):
            raise ValueError("truncated NVQ1-L custom codebook")
        codebook = unpack_ternary_codebook(blob[off : off + codebook_bytes])
        off += codebook_bytes
    ng = math.ceil(neuron_len / spec.groupsize)
    nvec = math.ceil(neuron_len / spec.vector_size)
    anchor_bytes = out * 2
    if off + anchor_bytes > len(blob):
        raise ValueError("truncated NVQ1-L neuron anchors")
    neuron_scale = np.frombuffer(
        blob,
        dtype=np.float16,
        count=out,
        offset=off,
    ).astype(np.float32)
    if not np.isfinite(neuron_scale).all() or np.signbit(neuron_scale).any():
        raise ValueError("NVQ1-L neuron anchors must be finite and non-negative")
    off += anchor_bytes
    sub_scale, off = _unpack_bits(blob, off, out * ng, spec.sub_bits)
    indices, off = _unpack_bits(blob, off, out * nvec, spec.index_bits)
    delta_sign, off = _unpack_bits(blob, off, out * ng, 1)
    if off != len(blob):
        raise ValueError(f"invalid NVQ1-L blob tail: consumed={off}, size={len(blob)}")

    return Nvq1LTensor(
        spec=spec,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale,
        sub_scale=sub_scale.astype(np.uint8).reshape(out, ng),
        indices=indices.reshape(out, nvec),
        delta_sign=delta_sign.astype(np.uint8).reshape(out, ng),
        codebook=codebook,
    )
