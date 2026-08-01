"""Packed OCP MXFP4/MXFP8 Metal kernels for Apple silicon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.mx import MXFP4_DTYPE, MxTensor, unpack_mx

_MX_HEADER = r"""
METAL_FUNC float mfq_mx_e8m0(uchar raw) {
    return raw == 255u ? NAN : exp2(float(int(raw) - 127));
}

METAL_FUNC float mfq_mx_fp4(uchar raw) {
    uchar magnitude = raw & 7u;
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (raw & 8u) == 0u ? value : -value;
}

METAL_FUNC float mfq_mx_fp8(uchar raw) {
    uint exponent = (uint(raw) >> 3u) & 15u;
    uint mantissa = uint(raw) & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return NAN;
    }
    float value = exponent == 0u
        ? ldexp(float(mantissa) * 0.125f, -6)
        : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
    return (raw & 128u) == 0u ? value : -value;
}

template <typename ValueStream, typename ScaleStream>
METAL_FUNC float mfq_mx_weight(
    ValueStream values,
    ScaleStream scales,
    uint output,
    uint column,
    uint mx_bits,
    uint width
) {
    if (mx_bits == 4u) {
        uchar packed = values[output * (width / 2u) + (column >> 1u)];
        uchar code = (column & 1u) == 0u ? packed & 15u : packed >> 4u;
        uchar scale = scales[output * (width / 32u) + column / 32u];
        return mfq_mx_fp4(code) * mfq_mx_e8m0(scale);
    }
    uchar code = values[output * width + column];
    uint scale_row = output / 128u;
    uint scale_column = column / 128u;
    uchar scale = scales[scale_row * (width / 128u) + scale_column];
    return mfq_mx_fp8(code) * mfq_mx_e8m0(scale);
}
"""

_MATMUL_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5u;
    uint output = workgroup % uint(OUT);
    uint first_row = (workgroup / uint(OUT)) * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }
    float accum[TILE_M];
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        accum[local] = 0.0f;
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        float weight = mfq_mx_weight(
            values, scales, output, column, uint(MX_BITS), uint(K));
        for (uint local = 0u; local < uint(TILE_M); ++local) {
            uint row = first_row + local;
            if (row < uint(M)) {
                accum[local] += float(x[row * uint(K) + column]) * weight;
            }
        }
    }
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
"""

_GEMV_SOURCE = r"""
    constexpr uint OUTPUTS_PER_SIMD = 4u;
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint OUTPUTS_PER_TG = OUTPUTS_PER_SIMD * SIMD_GROUPS;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint tg_index = thread_position_in_grid.x / 64u;
    uint first_output = tg_index * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    float accum[OUTPUTS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation = float(x[column]);
        for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
            uint output = first_output + local;
            if (output < uint(OUT)) {
                accum[local] += activation * mfq_mx_weight(
                    values, scales, output, column, uint(MX_BITS), uint(K));
            }
        }
    }
    for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
        uint output = first_output + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
"""

_DEQUANT_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint count = uint(OUT) * uint(K);
    if (index < count) {
        uint output = index / uint(K);
        uint column = index - output * uint(K);
        y[index] = T(mfq_mx_weight(
            values, scales, output, column, uint(MX_BITS), uint(K)));
    }
"""

_EMBEDDING_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint count = uint(M) * uint(K);
    if (index < count) {
        uint token = index / uint(K);
        uint column = index - token * uint(K);
        uint output = uint(ids[token]);
        y[index] = output < uint(OUT)
            ? T(mfq_mx_weight(
                values, scales, output, column, uint(MX_BITS), uint(K)))
            : T(NAN);
    }
"""

_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_packed_matmul",
    input_names=["values", "scales", "x"],
    output_names=["y"],
    source=_MATMUL_SOURCE,
    header=_MX_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)
_GEMV_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_packed_gemv",
    input_names=["values", "scales", "x"],
    output_names=["y"],
    source=_GEMV_SOURCE,
    header=_MX_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)
_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_dequantize",
    input_names=["values", "scales"],
    output_names=["y"],
    source=_DEQUANT_SOURCE,
    header=_MX_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)
_EMBEDDING_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_embedding",
    input_names=["values", "scales", "ids"],
    output_names=["y"],
    source=_EMBEDDING_SOURCE,
    header=_MX_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)


@dataclass(frozen=True)
class MetalMxWeight:
    """MX encoded values and E8M0 scales resident in unified Metal memory."""

    values: mx.array
    scales: mx.array
    bits: int
    out: int
    in_features: int

    @classmethod
    def from_tensor(cls, tensor: MxTensor) -> MetalMxWeight:
        return cls(
            values=mx.array(np.ascontiguousarray(tensor.values, dtype=np.uint8)),
            scales=mx.array(np.ascontiguousarray(tensor.scales, dtype=np.uint8)),
            bits=4 if tensor.dtype == MXFP4_DTYPE else 8,
            out=int(tensor.shape[0]),
            in_features=int(tensor.shape[1]),
        )

    @classmethod
    def from_blob(cls, dtype: str, blob: bytes | memoryview) -> MetalMxWeight:
        return cls.from_tensor(unpack_mx(dtype, blob))

    @property
    def packed_nbytes(self) -> int:
        return int(self.values.nbytes + self.scales.nbytes)


def _source_array(x: mx.array | np.ndarray) -> mx.array:
    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    return source


def _templates(
    weight: MetalMxWeight,
    dtype: mx.Dtype,
    *,
    rows: int = 1,
    tile_rows: int = 1,
) -> list[tuple[str, object]]:
    return [
        ("T", dtype),
        ("MX_BITS", weight.bits),
        ("K", weight.in_features),
        ("OUT", weight.out),
        ("M", rows),
        ("TILE_M", tile_rows),
    ]


def mx_dequantize(
    weight: MetalMxWeight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode an MX matrix, primarily for large-M dense GEMM and diagnostics."""

    if dtype not in (mx.float16, mx.float32):
        raise ValueError("MX dequantization requires float16 or float32")
    elements = weight.out * weight.in_features
    return _DEQUANT_KERNEL(
        inputs=[weight.values, weight.scales],
        template=_templates(weight, dtype),
        grid=(elements, 1, 1),
        threadgroup=(min(256, elements), 1, 1),
        output_shapes=[(weight.out, weight.in_features)],
        output_dtypes=[dtype],
    )[0]


def mx_matmul(weight: MetalMxWeight, x: mx.array | np.ndarray) -> mx.array:
    """Run packed GEMV/MMQ, or dequantize once for a large-M dense GEMM."""

    source = _source_array(x)
    if source.ndim == 0 or int(source.shape[-1]) != weight.in_features:
        raise ValueError(
            f"MX input width {source.shape if source.ndim else ()} does not match "
            f"packed width {weight.in_features}"
        )
    output_shape = tuple(int(value) for value in source.shape[:-1]) + (weight.out,)
    rows = int(source.size // weight.in_features)
    source = source.reshape(rows, weight.in_features)
    if rows >= 64:
        result = source @ mx_dequantize(weight, dtype=source.dtype).T
        return result.reshape(output_shape)
    gemv = rows == 1
    tile_rows = 1 if gemv else (rows if rows <= 16 else 8)
    row_tiles = (rows + tile_rows - 1) // tile_rows
    grid = ((weight.out + 7) // 8) * 64 if gemv else row_tiles * weight.out * 32
    kernel = _GEMV_KERNEL if gemv else _MATMUL_KERNEL
    result = kernel(
        inputs=[weight.values, weight.scales, source],
        template=_templates(weight, source.dtype, rows=rows, tile_rows=tile_rows),
        grid=(grid, 1, 1),
        threadgroup=(64 if gemv else 32, 1, 1),
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape(output_shape)


def mx_embedding(
    weight: MetalMxWeight,
    token_ids: mx.array | np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Look up rows from an MX encoded table."""

    ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    if dtype not in (mx.float16, mx.float32):
        raise ValueError("MX embedding requires float16 or float32 output")
    tokens = int(ids.size)
    output_shape = tuple(int(value) for value in ids.shape) + (weight.in_features,)
    if tokens == 0:
        return mx.zeros(output_shape, dtype=dtype)
    elements = tokens * weight.in_features
    return _EMBEDDING_KERNEL(
        inputs=[weight.values, weight.scales, ids.reshape(tokens)],
        template=_templates(weight, dtype, rows=tokens),
        grid=(elements, 1, 1),
        threadgroup=(min(256, elements), 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[dtype],
    )[0]


__all__ = [
    "MetalMxWeight",
    "mx_dequantize",
    "mx_embedding",
    "mx_matmul",
]
