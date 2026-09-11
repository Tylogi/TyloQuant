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

_BACKWARD_INPUT_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint column = threadgroup_position_in_grid.x;
    uint first_row = threadgroup_position_in_grid.y * uint(TILE_M);
    if (column >= uint(K) || first_row >= uint(M)) {
        return;
    }
    float accum[TILE_M];
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        accum[local] = 0.0f;
    }
    for (uint output = lane; output < uint(OUT); output += 32u) {
        float weight = mfq_mx_weight(
            values, scales, output, column, uint(MX_BITS), uint(K));
        for (uint local = 0u; local < uint(TILE_M); ++local) {
            uint row = first_row + local;
            if (row < uint(M)) {
                accum[local] = fma(
                    float(x[row * uint(OUT) + output]), weight, accum[local]);
            }
        }
    }
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && row < uint(M)) {
            y[row * uint(K) + column] = T(total);
        }
    }
"""

_BACKWARD_QUAD_SOURCE = r"""
    uint quad = thread_position_in_grid.x;
    uint column = quad * 4u;
    uint first_row = threadgroup_position_in_grid.y * uint(TILE_M);
    if (column >= uint(K) || first_row >= uint(M)) {
        return;
    }
    float4 accumulators[TILE_M];
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        accumulators[local] = float4(0.0f);
    }
    for (uint output = 0u; output < uint(OUT); ++output) {
        float4 weights;
        if (MX_BITS == 4) {
            uint packed_row = output * (uint(K) >> 1u);
            uchar packed0 = values[packed_row + (column >> 1u)];
            uchar packed1 = values[packed_row + (column >> 1u) + 1u];
            uchar scale = scales[
                output * (uint(K) >> 5u) + (column >> 5u)
            ];
            float multiplier = mfq_mx_e8m0(scale);
            weights = multiplier * float4(
                mfq_mx_fp4(packed0 & 15u),
                mfq_mx_fp4(packed0 >> 4u),
                mfq_mx_fp4(packed1 & 15u),
                mfq_mx_fp4(packed1 >> 4u));
        } else {
            uint value_offset = output * uint(K) + column;
            uint scale_row = output >> 7u;
            uint scale_column = column >> 7u;
            float multiplier = mfq_mx_e8m0(
                scales[scale_row * (uint(K) >> 7u) + scale_column]
            );
            uchar4 codes = *((const device uchar4*)(values + value_offset));
            weights = multiplier * float4(
                mfq_mx_fp8(codes.x),
                mfq_mx_fp8(codes.y),
                mfq_mx_fp8(codes.z),
                mfq_mx_fp8(codes.w));
        }
        for (uint local = 0u; local < uint(TILE_M); ++local) {
            uint row = first_row + local;
            if (row < uint(M)) {
                float gradient = float(x[row * uint(OUT) + output]);
                accumulators[local] = fma(
                    float4(gradient),
                    weights,
                    accumulators[local]);
            }
        }
    }
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        if (row < uint(M)) {
            for (uint component = 0u; component < 4u; ++component) {
                if (column + component < uint(K)) {
                    y[row * uint(K) + column + component] =
                        T(accumulators[local][component]);
                }
            }
        }
    }
"""

_BACKWARD_MATRIX_SOURCE = r"""
    constexpr uint BM = 8u;
    constexpr uint BN = 64u;
    constexpr uint BK = 128u;
    constexpr uint BN_PAD = BN + 8u;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * BM;
    uint column_base = threadgroup_position_in_grid.x * BK;
    threadgroup half gradient_tile[BM * BN_PAD];
    threadgroup half weight_tile[BN * BK];
    metal::simdgroup_matrix<float, 8, 8> c0;
    metal::simdgroup_matrix<float, 8, 8> c1;
    c0.thread_elements()[0] = 0.0f;
    c0.thread_elements()[1] = 0.0f;
    c1.thread_elements()[0] = 0.0f;
    c1.thread_elements()[1] = 0.0f;
    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint simd_col = simd_group * 16u;
    uint chunks = (uint(OUT) + BN - 1u) / BN;
    for (uint chunk = 0u; chunk < chunks; ++chunk) {
        uint output_base = chunk * BN;
        for (uint index = local_thread; index < BM * BN; index += 256u) {
            uint local_row = index / BN;
            uint local_output = index - local_row * BN;
            uint row = row_base + local_row;
            uint output = output_base + local_output;
            gradient_tile[local_row * BN_PAD + local_output] =
                row < uint(M) && output < uint(OUT)
                ? half(x[row * uint(OUT) + output])
                : half(0.0f);
        }
        for (uint index = local_thread; index < BN * BK; index += 256u) {
            uint local_output = index / BK;
            uint local_column = index - local_output * BK;
            uint output = output_base + local_output;
            uint column = column_base + local_column;
            float value = output < uint(OUT) && column < uint(K)
                ? mfq_mx_weight(
                    values,
                    scales,
                    output,
                    column,
                    uint(MX_BITS),
                    uint(K))
                : 0.0f;
            weight_tile[local_output * BK + local_column] = half(value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BN; kk += 8u) {
            metal::simdgroup_matrix<half, 8, 8> a;
            metal::simdgroup_matrix<half, 8, 8> b0;
            metal::simdgroup_matrix<half, 8, 8> b1;
            a.thread_elements()[0] = gradient_tile[
                fragment_row * BN_PAD + kk + fragment_col];
            a.thread_elements()[1] = gradient_tile[
                fragment_row * BN_PAD + kk + fragment_col + 1u];
            b0.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BK + simd_col + fragment_col];
            b0.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BK + simd_col + fragment_col + 1u];
            b1.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BK + simd_col + 8u + fragment_col];
            b1.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BK + simd_col + 8u + fragment_col + 1u];
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row = row_base + fragment_row;
    uint local_col0 = simd_col + fragment_col;
    uint local_col1 = local_col0 + 8u;
    uint col0 = column_base + local_col0;
    uint col1 = column_base + local_col1;
    if (row < uint(M)) {
        if (col0 < uint(K)) {
            y[row * uint(K) + col0] = T(c0.thread_elements()[0]);
        }
        if (col0 + 1u < uint(K)) {
            y[row * uint(K) + col0 + 1u] = T(c0.thread_elements()[1]);
        }
        if (col1 < uint(K)) {
            y[row * uint(K) + col1] = T(c1.thread_elements()[0]);
        }
        if (col1 + 1u < uint(K)) {
            y[row * uint(K) + col1 + 1u] = T(c1.thread_elements()[1]);
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
_BACKWARD_INPUT_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_packed_backward_input",
    input_names=["values", "scales", "x"],
    output_names=["y"],
    source=_BACKWARD_INPUT_SOURCE,
    header=_MX_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)
_BACKWARD_QUAD_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_packed_backward_quad",
    input_names=["values", "scales", "x"],
    output_names=["y"],
    source=_BACKWARD_QUAD_SOURCE,
    header=_MX_HEADER,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "fast"},
)
_BACKWARD_MATRIX_KERNEL = mx.fast.metal_kernel(
    name="mfq_mx_packed_backward_matrix",
    input_names=["values", "scales", "x"],
    output_names=["y"],
    source=_BACKWARD_MATRIX_SOURCE,
    header="#include <metal_simdgroup_matrix>\n" + _MX_HEADER,
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


def _mx_matmul_impl(weight: MetalMxWeight, source: mx.array) -> mx.array:
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


def mx_backward_input(
    weight: MetalMxWeight,
    output_gradient: mx.array | np.ndarray,
) -> mx.array:
    """Compute ``dX = dY @ W`` directly from packed MXFP4/MXFP8 storage."""

    gradient = _source_array(output_gradient)
    if gradient.ndim == 0 or int(gradient.shape[-1]) != weight.out:
        raise ValueError(
            f"MX output-gradient width must be {weight.out}, got "
            f"{gradient.shape if gradient.ndim else ()}"
        )
    prefix = tuple(int(value) for value in gradient.shape[:-1])
    rows = int(gradient.size) // weight.out
    if rows == 0:
        return mx.zeros((*prefix, weight.in_features), dtype=gradient.dtype)
    gradient = mx.contiguous(gradient.reshape((rows, weight.out)))
    if gradient.dtype != mx.float16:
        tile_rows = min(rows, 4)
        result = _BACKWARD_INPUT_KERNEL(
            inputs=[weight.values, weight.scales, gradient],
            template=_templates(
                weight,
                gradient.dtype,
                rows=rows,
                tile_rows=tile_rows,
            ),
            grid=(
                weight.in_features * 32,
                (rows + tile_rows - 1) // tile_rows,
                1,
            ),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, weight.in_features)],
            output_dtypes=[gradient.dtype],
        )[0]
        return result.reshape((*prefix, weight.in_features))
    if rows > 8:
        dense = mx_dequantize(weight, dtype=gradient.dtype)
        return (gradient @ dense).reshape((*prefix, weight.in_features))
    if rows >= 5:
        result = _BACKWARD_MATRIX_KERNEL(
            inputs=[weight.values, weight.scales, gradient],
            template=_templates(weight, gradient.dtype, rows=rows, tile_rows=8),
            grid=(
                ((weight.in_features + 127) // 128) * 256,
                (rows + 7) // 8,
                1,
            ),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, weight.in_features)],
            output_dtypes=[gradient.dtype],
        )[0]
        return result.reshape((*prefix, weight.in_features))
    tile_rows = min(rows, 4)
    result = _BACKWARD_QUAD_KERNEL(
        inputs=[weight.values, weight.scales, gradient],
        template=_templates(weight, gradient.dtype, rows=rows, tile_rows=tile_rows),
        grid=(
            ((weight.in_features + 127) // 128) * 32,
            (rows + tile_rows - 1) // tile_rows,
            1,
        ),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, weight.in_features)],
        output_dtypes=[gradient.dtype],
    )[0]
    return result.reshape((*prefix, weight.in_features))


def mx_matmul(weight: MetalMxWeight, x: mx.array | np.ndarray) -> mx.array:
    """Run MX matmul with a direct packed custom VJP for its input."""

    source = _source_array(x)

    @mx.custom_function
    def operation(value: mx.array) -> mx.array:
        return _mx_matmul_impl(weight, value)

    @operation.vjp
    def operation_vjp(primals, cotangent, output):
        del primals, output
        return mx_backward_input(weight, cotangent)

    return operation(source)


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
    "mx_backward_input",
    "mx_dequantize",
    "mx_embedding",
    "mx_matmul",
]
