"""GGML-compatible NINT8-0 Metal kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.nint8_zero import Nint8ZeroTensor, unpack_nint8_zero

_MATMUL_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint first_row = (workgroup / uint(OUT)) * uint(TILE_M);
    if (first_row >= uint(M) || output >= uint(OUT)) {
        return;
    }

    float accumulators[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        accumulators[row] = 0.0f;
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        uint group = column >> 5;
        float weight = float(scales[output * uint(NG) + group])
            * float(q[output * uint(K) + column]);
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] +=
                    float(x[row * uint(K) + column]) * weight;
            }
        }
    }
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
"""

_GEMV_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint output = thread_position_in_grid.x >> 5;
    if (output >= uint(OUT)) {
        return;
    }

    float accumulator = 0.0f;
    for (uint column = lane * 4u;
         column < uint(K);
         column += 128u) {
        uint group = column >> 5;
        float scale = float(scales[output * uint(NG) + group]);
        uint offset = output * uint(K) + column;
        const device char4* packed =
            (const device char4*)(q + offset);
        char4 codes = *packed;
        float4 activations = float4(
            float(x[column]),
            float(x[column + 1u]),
            float(x[column + 2u]),
            float(x[column + 3u]));
        float4 weights = scale * float4(codes);
        accumulator +=
            activations.x * weights.x
            + activations.y * weights.y
            + activations.z * weights.z
            + activations.w * weights.w;
    }

    float total = simd_sum(accumulator);
    if (lane == 0u) {
        y[output] = T(total);
    }
"""

_GEMM_MATRIX_SOURCE = r"""
    constexpr uint BM = 32u;
    constexpr uint BN = 64u;
    constexpr uint GS = 32u;
    constexpr uint GPC = 3u;
    constexpr uint BK = GS * GPC;
    constexpr uint BK_PAD = BK + 8u;
    constexpr uint BN_PAD = BN + 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * BM;
    uint output_base = threadgroup_position_in_grid.x * BN;

    threadgroup half activation_tile[BM * BK_PAD];
    threadgroup half weight_tile[BK * BN_PAD];

    metal::simdgroup_matrix<float, 8, 8> c00;
    metal::simdgroup_matrix<float, 8, 8> c01;
    metal::simdgroup_matrix<float, 8, 8> c10;
    metal::simdgroup_matrix<float, 8, 8> c11;
    c00.thread_elements()[0] = 0.0f;
    c00.thread_elements()[1] = 0.0f;
    c01.thread_elements()[0] = 0.0f;
    c01.thread_elements()[1] = 0.0f;
    c10.thread_elements()[0] = 0.0f;
    c10.thread_elements()[1] = 0.0f;
    c11.thread_elements()[0] = 0.0f;
    c11.thread_elements()[1] = 0.0f;

    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint simd_row = (simd_group / 4u) * 16u;
    uint simd_col = (simd_group & 3u) * 16u;

    uint chunks = (uint(NG) + GPC - 1u) / GPC;
    for (uint chunk = 0u; chunk < chunks; ++chunk) {
        uint group_base = chunk * GPC;
        uint column_base = group_base * GS;

        for (
            uint index = local_thread;
            index < BM * BK;
            index += 256u
        ) {
            uint local_row = index / BK;
            uint local_column = index - local_row * BK;
            uint row = row_base + local_row;
            uint column = column_base + local_column;
            activation_tile[local_row * BK_PAD + local_column] =
                row < uint(M) && column < uint(K)
                ? x[row * uint(K) + column]
                : half(0.0f);
        }

        // Match the CUDA common FP16 MMQ loader: one thread expands one
        // complete Q8_0 output/group pair, then all eight SIMD-groups reuse
        // the transient tile.
        for (
            uint task = local_thread;
            task < BN * GPC;
            task += 256u
        ) {
            uint local_output = task / GPC;
            uint local_group = task - local_output * GPC;
            uint output = output_base + local_output;
            uint group = group_base + local_group;
            bool valid = output < uint(OUT) && group < uint(NG);
            uint metadata_index = output * uint(NG) + group;
            float scale = valid ? float(scales[metadata_index]) : 0.0f;
            for (uint element = 0u; element < GS; ++element) {
                uint local_column = local_group * GS + element;
                uint column = column_base + local_column;
                float value = valid && column < uint(K)
                    ? scale * float(q[metadata_index * GS + element])
                    : 0.0f;
                weight_tile[
                    local_column * BN_PAD + local_output
                ] = half(value);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK; kk += 8u) {
            metal::simdgroup_matrix<half, 8, 8> a0;
            metal::simdgroup_matrix<half, 8, 8> a1;
            metal::simdgroup_matrix<half, 8, 8> b0;
            metal::simdgroup_matrix<half, 8, 8> b1;

            a0.thread_elements()[0] = activation_tile[
                (simd_row + fragment_row) * BK_PAD + kk + fragment_col];
            a0.thread_elements()[1] = activation_tile[
                (simd_row + fragment_row) * BK_PAD + kk + fragment_col + 1u];
            a1.thread_elements()[0] = activation_tile[
                (simd_row + 8u + fragment_row) * BK_PAD + kk + fragment_col];
            a1.thread_elements()[1] = activation_tile[
                (simd_row + 8u + fragment_row) * BK_PAD
                + kk + fragment_col + 1u];

            b0.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BN_PAD + simd_col + fragment_col];
            b0.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BN_PAD + simd_col + fragment_col + 1u];
            b1.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BN_PAD
                + simd_col + 8u + fragment_col];
            b1.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BN_PAD
                + simd_col + 8u + fragment_col + 1u];

            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint row0 = row_base + simd_row + fragment_row;
    uint row1 = row0 + 8u;
    uint col0 = output_base + simd_col + fragment_col;
    uint col1 = col0 + 8u;
    if (row0 < uint(M)) {
        if (col0 < uint(OUT)) {
            y[row0 * uint(OUT) + col0] = half(c00.thread_elements()[0]);
        }
        if (col0 + 1u < uint(OUT)) {
            y[row0 * uint(OUT) + col0 + 1u] =
                half(c00.thread_elements()[1]);
        }
        if (col1 < uint(OUT)) {
            y[row0 * uint(OUT) + col1] = half(c01.thread_elements()[0]);
        }
        if (col1 + 1u < uint(OUT)) {
            y[row0 * uint(OUT) + col1 + 1u] =
                half(c01.thread_elements()[1]);
        }
    }
    if (row1 < uint(M)) {
        if (col0 < uint(OUT)) {
            y[row1 * uint(OUT) + col0] = half(c10.thread_elements()[0]);
        }
        if (col0 + 1u < uint(OUT)) {
            y[row1 * uint(OUT) + col0 + 1u] =
                half(c10.thread_elements()[1]);
        }
        if (col1 < uint(OUT)) {
            y[row1 * uint(OUT) + col1] = half(c11.thread_elements()[0]);
        }
        if (col1 + 1u < uint(OUT)) {
            y[row1 * uint(OUT) + col1 + 1u] =
                half(c11.thread_elements()[1]);
        }
    }
"""

_DEQUANT_SOURCE = r"""
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(OUT * K)) {
        return;
    }
    uint output = linear / uint(K);
    uint column = linear - output * uint(K);
    uint group = column >> 5;
    y[linear] = T(
        float(scales[output * uint(NG) + group])
        * float(q[linear])
    );
"""

_EMBEDDING_SOURCE = r"""
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(COUNT * K)) {
        return;
    }
    uint token = linear / uint(K);
    uint column = linear - token * uint(K);
    uint output = uint(token_ids[token]);
    uint group = column >> 5;
    y[linear] = T(
        float(scales[output * uint(NG) + group])
        * float(q[output * uint(K) + column])
    );
"""

_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint8_zero_matmul",
    input_names=["q", "scales", "x"],
    output_names=["y"],
    source=_MATMUL_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GEMV_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint8_zero_gemv_vec4",
    input_names=["q", "scales", "x"],
    output_names=["y"],
    source=_GEMV_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GEMM_MATRIX_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint8_zero_gemm_matrix",
    input_names=["q", "scales", "x"],
    output_names=["y"],
    source=_GEMM_MATRIX_SOURCE,
    compile_options={"math_mode": "fast"},
)

_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint8_zero_dequant",
    input_names=["q", "scales"],
    output_names=["y"],
    source=_DEQUANT_SOURCE,
    compile_options={"math_mode": "fast"},
)

_EMBEDDING_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint8_zero_embedding",
    input_names=["q", "scales", "token_ids"],
    output_names=["y"],
    source=_EMBEDDING_SOURCE,
    compile_options={"math_mode": "fast"},
)


@dataclass(frozen=True)
class MetalNint8ZeroWeight:
    """Execution-ready Q8_0 blocks resident in MLX/Metal memory."""

    q: mx.array
    scales: mx.array
    out: int
    groups: int
    neuron_len: int

    @classmethod
    def from_tensor(cls, tensor: Nint8ZeroTensor) -> MetalNint8ZeroWeight:
        if len(tensor.shape) != 2 or tensor.axis != 0:
            raise ValueError("Metal NINT8-0 currently requires a 2D weight quantized on axis 0")
        return cls(
            q=mx.array(np.ascontiguousarray(tensor.q, dtype=np.int8).reshape(-1)),
            scales=mx.array(np.ascontiguousarray(tensor.scale, dtype=np.float16).reshape(-1)),
            out=tensor.out,
            groups=tensor.ng,
            neuron_len=tensor.neuron_len,
        )

    @classmethod
    def from_blob(
        cls,
        blob: bytes | memoryview,
    ) -> MetalNint8ZeroWeight:
        return cls.from_tensor(unpack_nint8_zero(blob))

    @property
    def packed_nbytes(self) -> int:
        return int(self.q.nbytes) + int(self.scales.nbytes)


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def _prepare(
    weight: MetalNint8ZeroWeight,
    x: mx.array | np.ndarray,
) -> tuple[mx.array, tuple[int, ...], int]:
    source = _floating(x)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError(f"NINT8-0 input must end in width {weight.neuron_len}")
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    return source.reshape((rows, weight.neuron_len)), prefix, rows


def _simple_matmul(
    weight: MetalNint8ZeroWeight,
    source: mx.array,
    prefix: tuple[int, ...],
    rows: int,
    *,
    tile_rows: int,
) -> mx.array:
    row_tiles = (rows + tile_rows - 1) // tile_rows
    result = _MATMUL_KERNEL(
        inputs=[weight.q, weight.scales, source],
        template=[
            ("T", source.dtype),
            ("M", rows),
            ("TILE_M", tile_rows),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("NG", weight.groups),
        ],
        grid=(row_tiles * weight.out * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape((*prefix, weight.out))


def nint8_zero_gemv(
    weight: MetalNint8ZeroWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Single-row packed Q8_0 matrix-vector multiply."""

    source, prefix, rows = _prepare(weight, x)
    if rows != 1:
        raise ValueError("NINT8-0 GEMV requires exactly one input row")
    result = _GEMV_KERNEL(
        inputs=[weight.q, weight.scales, source],
        template=[
            ("T", source.dtype),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("NG", weight.groups),
        ],
        grid=(weight.out * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, weight.out)],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape((*prefix, weight.out))


def nint8_zero_mmq(
    weight: MetalNint8ZeroWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Small-M packed Q8_0 multiply reusing weights across input rows."""

    source, prefix, rows = _prepare(weight, x)
    if not 2 <= rows <= 16:
        raise ValueError("NINT8-0 MMQ requires 2 to 16 input rows")
    return _simple_matmul(
        weight,
        source,
        prefix,
        rows,
        tile_rows=min(8, rows),
    )


def nint8_zero_gemm(
    weight: MetalNint8ZeroWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Online-decode Q8_0 GEMM using Apple simdgroup matrix operations."""

    source, prefix, rows = _prepare(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, weight.out), dtype=source.dtype)
    if source.dtype != mx.float16:
        return _simple_matmul(
            weight,
            source,
            prefix,
            rows,
            tile_rows=min(8, rows),
        )
    result = _GEMM_MATRIX_KERNEL(
        inputs=[weight.q, weight.scales, source],
        template=[
            ("M", rows),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("NG", weight.groups),
        ],
        grid=(
            ((weight.out + 63) // 64) * 256,
            (rows + 31) // 32,
            1,
        ),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape((*prefix, weight.out))


def nint8_zero_packed_matmul(
    weight: MetalNint8ZeroWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Dispatch Q8_0 GEMV, small-M MMQ, or online-decode packed GEMM."""

    source, prefix, rows = _prepare(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, weight.out), dtype=source.dtype)
    if rows == 1:
        return nint8_zero_gemv(weight, x)
    if rows <= 16:
        return nint8_zero_mmq(weight, x)
    return nint8_zero_gemm(weight, x)


def nint8_zero_dequantize(
    weight: MetalNint8ZeroWeight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode a Q8_0 matrix to a temporary dense Metal array."""

    size = weight.out * weight.neuron_len
    return _DEQUANT_KERNEL(
        inputs=[weight.q, weight.scales],
        template=[
            ("T", dtype),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("NG", weight.groups),
        ],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(weight.out, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]


def nint8_zero_matmul(
    weight: MetalNint8ZeroWeight,
    x: mx.array | np.ndarray,
    *,
    dequantize_threshold: int | None = 64,
) -> mx.array:
    """Dispatch Q8_0 matmul across packed and temporary-dense paths."""

    source = x if isinstance(x, mx.array) else mx.array(x)
    rows = int(np.prod(tuple(int(value) for value in source.shape[:-1]))) if source.ndim > 1 else 1
    if (
        dequantize_threshold is not None
        and rows >= int(dequantize_threshold)
        and source.dtype == mx.float16
    ):
        prepared, prefix, _ = _prepare(weight, source)
        dense = nint8_zero_dequantize(weight, dtype=mx.float16)
        return (prepared @ dense.T).reshape((*prefix, weight.out))
    return nint8_zero_packed_matmul(weight, source)


def nint8_zero_embedding(
    weight: MetalNint8ZeroWeight,
    token_ids: mx.array | np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode only selected Q8_0 rows."""

    ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    shape = tuple(int(value) for value in ids.shape)
    count = int(ids.size)
    if count == 0:
        return mx.zeros((*shape, weight.neuron_len), dtype=dtype)
    result = _EMBEDDING_KERNEL(
        inputs=[weight.q, weight.scales, mx.contiguous(ids.reshape((-1,)))],
        template=[
            ("T", dtype),
            ("COUNT", count),
            ("K", weight.neuron_len),
            ("NG", weight.groups),
        ],
        grid=(count * weight.neuron_len, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(count, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]
    return result.reshape((*shape, weight.neuron_len))


__all__ = [
    "MetalNint8ZeroWeight",
    "nint8_zero_dequantize",
    "nint8_zero_embedding",
    "nint8_zero_gemm",
    "nint8_zero_gemv",
    "nint8_zero_matmul",
    "nint8_zero_mmq",
    "nint8_zero_packed_matmul",
]
