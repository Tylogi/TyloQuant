"""General-purpose Transformer kernels for Apple silicon.

These MLX custom Metal kernels cover operations shared by Llama, Qwen, Gemma,
and related decoder-only architectures. Inputs are contiguous float16 or
float32 arrays; reductions accumulate in float32.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_ELEMENTWISE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }

    float av = float(a[index]);
    float bv = float(b[index]);
    float value;
    if (OP == 0) {
        value = av + bv;
    } else if (OP == 1) {
        value = (av / (1.0f + exp(-av))) * bv;
    } else if (OP == 2) {
        constexpr float gelu_scale = 0.7978845608028654f;
        float inner = gelu_scale * (av + 0.044715f * av * av * av);
        value = 0.5f * av * (1.0f + tanh(inner)) * bv;
    } else {
        value = av * bv;
    }
    y[index] = T(value);
"""


_RMS_NORM_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }

    uint offset = row * uint(DIM);
    float square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        float value = float(x[offset + column]);
        square_sum += value * value;
    }
    square_sum = simd_sum(square_sum);
    float inverse = rsqrt(square_sum / float(DIM) + params[0]);
    float weight_offset = params[1];

    for (uint column = lane; column < uint(DIM); column += 32u) {
        float value = float(x[offset + column]);
        y[offset + column] = T(value * inverse * (weight[column] + weight_offset));
    }
"""


_L2_NORM_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }

    uint offset = row * uint(DIM);
    float square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        float value = float(x[offset + column]);
        square_sum += value * value;
    }
    square_sum = simd_sum(square_sum);
    float inverse = 1.0f / max(sqrt(square_sum), params[0]);

    for (uint column = lane; column < uint(DIM); column += 32u) {
        y[offset + column] = T(float(x[offset + column]) * inverse);
    }
"""


_RESIDUAL_RMS_NORM_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }

    uint offset = row * uint(DIM);
    float square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        T stored = T(float(a[offset + column]) + float(b[offset + column]));
        residual[offset + column] = stored;
        float value = float(stored);
        square_sum += value * value;
    }
    square_sum = simd_sum(square_sum);
    float inverse = rsqrt(square_sum / float(DIM) + params[0]);
    float weight_offset = params[1];

    for (uint column = lane; column < uint(DIM); column += 32u) {
        T stored = T(float(a[offset + column]) + float(b[offset + column]));
        float value = float(stored) * inverse * (weight[column] + weight_offset);
        normalized[offset + column] = O(value);
    }
"""


_ROPE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }

    uint column = index % uint(DIM);
    if (column >= uint(ROTARY_DIM)) {
        y[index] = x[index];
        return;
    }

    constexpr uint half_width = ROTARY_DIM / 2;
    uint row = index / uint(DIM);
    uint token = row % uint(TOKENS);
    uint pair = column < half_width ? column : column - half_width;
    uint axis = 0u;
    if (S0 + S1 + S2 > 0) {
        axis = pair < uint(S0) ? 0u : (pair < uint(S0 + S1) ? 1u : 2u);
        if (axis >= uint(POS_AXES)) {
            axis = 0u;
        }
    }

    int position = positions[(POS_AXES == 1 ? 0u : axis) * uint(TOKENS) + token];
    position = max(0, min(position, TABLE_LEN - 1));
    uint table_index = uint(position) * half_width + pair;
    float cosine = cos_table[table_index];
    float sine = sin_table[table_index];
    uint row_offset = row * uint(DIM);
    float first = float(x[row_offset + pair]);
    float second = float(x[row_offset + pair + half_width]);
    float value = column < half_width
        ? first * cosine - second * sine
        : second * cosine + first * sine;
    y[index] = T(value);
"""


_ELEMENTWISE_KERNEL = mx.fast.metal_kernel(
    name="mfq_transformer_elementwise",
    input_names=["a", "b"],
    output_names=["y"],
    source=_ELEMENTWISE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_RMS_NORM_KERNEL = mx.fast.metal_kernel(
    name="mfq_rms_norm",
    input_names=["x", "weight", "params"],
    output_names=["y"],
    source=_RMS_NORM_SOURCE,
    compile_options={"math_mode": "fast"},
)

_L2_NORM_KERNEL = mx.fast.metal_kernel(
    name="mfq_l2_norm",
    input_names=["x", "params"],
    output_names=["y"],
    source=_L2_NORM_SOURCE,
    compile_options={"math_mode": "fast"},
)

_RESIDUAL_RMS_NORM_KERNEL = mx.fast.metal_kernel(
    name="mfq_residual_rms_norm",
    input_names=["a", "b", "weight", "params"],
    output_names=["residual", "normalized"],
    source=_RESIDUAL_RMS_NORM_SOURCE,
    compile_options={"math_mode": "fast"},
)

_ROPE_KERNEL = mx.fast.metal_kernel(
    name="mfq_rotate_half_rope",
    input_names=["x", "positions", "cos_table", "sin_table"],
    output_names=["y"],
    source=_ROPE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_ROPE_TABLES: dict[
    tuple[float, int, int, int, int],
    tuple[mx.array, mx.array],
] = {}


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def _matching_floating(
    a: mx.array | np.ndarray,
    b: mx.array | np.ndarray,
) -> tuple[mx.array, mx.array]:
    left = _floating(a)
    right = _floating(b)
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Metal elementwise input shapes differ: {left.shape} != {right.shape}")
    dtype = mx.float32 if mx.float32 in (left.dtype, right.dtype) else mx.float16
    return mx.contiguous(left.astype(dtype)), mx.contiguous(right.astype(dtype))


def _threadgroup_size(size: int) -> tuple[int, int, int]:
    return (min(256, max(1, int(size))), 1, 1)


def _elementwise(
    a: mx.array | np.ndarray,
    b: mx.array | np.ndarray,
    operation: int,
) -> mx.array:
    left, right = _matching_floating(a, b)
    size = int(left.size)
    if size == 0:
        return mx.zeros(left.shape, dtype=left.dtype)
    return _ELEMENTWISE_KERNEL(
        inputs=[left, right],
        template=[("T", left.dtype), ("SIZE", size), ("OP", int(operation))],
        grid=(size, 1, 1),
        threadgroup=_threadgroup_size(size),
        output_shapes=[left.shape],
        output_dtypes=[left.dtype],
    )[0]


def residual_add(a: mx.array | np.ndarray, b: mx.array | np.ndarray) -> mx.array:
    """Add two residual-stream arrays with float16/float32 promotion."""

    return _elementwise(a, b, 0)


def silu_mul(gate: mx.array | np.ndarray, up: mx.array | np.ndarray) -> mx.array:
    """Compute ``silu(gate) * up`` in one Metal pass."""

    return _elementwise(gate, up, 1)


def gelu_mul(gate: mx.array | np.ndarray, up: mx.array | np.ndarray) -> mx.array:
    """Compute tanh-approximated ``gelu(gate) * up`` in one Metal pass."""

    return _elementwise(gate, up, 2)


def hadamard_mul(a: mx.array | np.ndarray, b: mx.array | np.ndarray) -> mx.array:
    """Multiply two arrays elementwise in one Metal pass."""

    return _elementwise(a, b, 3)


def _norm_inputs(
    x: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
) -> tuple[mx.array, mx.array, tuple[int, ...], int, int]:
    source = _floating(x)
    if source.ndim < 1:
        raise ValueError("Metal norm input must have at least one dimension")
    dimension = int(source.shape[-1])
    if dimension <= 0:
        raise ValueError("Metal norm width must be positive")
    scale = weight if isinstance(weight, mx.array) else mx.array(weight)
    if scale.ndim != 1 or int(scale.size) != dimension:
        raise ValueError(f"RMSNorm weight must have shape ({dimension},), got {scale.shape}")
    scale = mx.contiguous(scale.astype(mx.float32))
    shape = tuple(int(value) for value in source.shape)
    rows = int(source.size) // dimension
    return source, scale, shape, rows, dimension


def rms_norm(
    x: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
    eps: float = 1e-6,
    *,
    weight_offset: float = 0.0,
) -> mx.array:
    """RMS-normalize the final dimension with float32 accumulation."""

    source, scale, shape, rows, dimension = _norm_inputs(x, weight)
    if rows == 0:
        return mx.zeros(shape, dtype=source.dtype)
    params = mx.array([float(eps), float(weight_offset)], dtype=mx.float32)
    grid_size = rows * 32
    return _RMS_NORM_KERNEL(
        inputs=[source, scale, params],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("DIM", dimension),
        ],
        grid=(grid_size, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape],
        output_dtypes=[source.dtype],
    )[0]


def l2_norm(x: mx.array | np.ndarray, eps: float = 1e-5) -> mx.array:
    """L2-normalize the final dimension using ``max(norm, eps)``."""

    source = _floating(x)
    if source.ndim < 1:
        raise ValueError("Metal L2 norm input must have at least one dimension")
    shape = tuple(int(value) for value in source.shape)
    dimension = int(shape[-1])
    if dimension <= 0:
        raise ValueError("Metal L2 norm width must be positive")
    rows = int(source.size) // dimension
    if rows == 0:
        return mx.zeros(shape, dtype=source.dtype)
    params = mx.array([float(eps)], dtype=mx.float32)
    return _L2_NORM_KERNEL(
        inputs=[source, params],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("DIM", dimension),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape],
        output_dtypes=[source.dtype],
    )[0]


def residual_rms_norm(
    a: mx.array | np.ndarray,
    b: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
    eps: float = 1e-6,
    *,
    weight_offset: float = 0.0,
    normalized_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Fuse residual addition and RMSNorm, returning ``(sum, normalized)``."""

    left, right = _matching_floating(a, b)
    source, scale, shape, rows, dimension = _norm_inputs(left, weight)
    output_dtype = source.dtype if normalized_dtype is None else normalized_dtype
    if output_dtype not in (mx.float16, mx.float32):
        raise ValueError("normalized_dtype must be mlx.float16 or mlx.float32")
    if rows == 0:
        return (
            mx.zeros(shape, dtype=source.dtype),
            mx.zeros(shape, dtype=output_dtype),
        )
    params = mx.array([float(eps), float(weight_offset)], dtype=mx.float32)
    return tuple(
        _RESIDUAL_RMS_NORM_KERNEL(
            inputs=[source, right, scale, params],
            template=[
                ("T", source.dtype),
                ("O", output_dtype),
                ("ROWS", rows),
                ("DIM", dimension),
            ],
            grid=(rows * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[shape, shape],
            output_dtypes=[source.dtype, output_dtype],
        )
    )


def rope_tables(
    base: float,
    rotary_dim: int,
    table_len: int,
    *,
    frequency_dim: int | None = None,
    active_pairs: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Return cached float32 rotate-half RoPE tables.

    ``frequency_dim`` controls the exponent denominator independently from
    the physical rotate-half width.  ``active_pairs`` leaves the remaining
    pairs as identity rotations, matching Gemma4 partial full-attention RoPE.
    """

    rotary_dim = int(rotary_dim)
    table_len = int(table_len)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    if table_len <= 0:
        raise ValueError("table_len must be positive")
    denominator = rotary_dim if frequency_dim is None else int(frequency_dim)
    if denominator <= 0:
        raise ValueError("frequency_dim must be positive")
    pairs = rotary_dim // 2
    active = pairs if active_pairs is None else int(active_pairs)
    if not 0 <= active <= pairs:
        raise ValueError("active_pairs must be in [0, rotary_dim / 2]")
    key = (float(base), rotary_dim, table_len, denominator, active)
    cached = _ROPE_TABLES.get(key)
    if cached is not None:
        return cached
    pair_indices = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    frequency = mx.power(
        mx.array(float(base), dtype=mx.float32),
        -pair_indices / float(denominator),
    )
    if active < pairs:
        frequency = mx.where(
            mx.arange(pairs, dtype=mx.int32) < active,
            frequency,
            mx.zeros((pairs,), dtype=mx.float32),
        )
    angles = mx.arange(table_len, dtype=mx.float32)[:, None] * frequency[None, :]
    cached = (mx.contiguous(mx.cos(angles)), mx.contiguous(mx.sin(angles)))
    mx.eval(*cached)
    _ROPE_TABLES[key] = cached
    return cached


def _position_array(positions: mx.array | np.ndarray) -> mx.array:
    result = positions if isinstance(positions, mx.array) else mx.array(positions)
    if result.ndim not in (1, 2):
        raise ValueError("RoPE positions must have [T] or [axes, T] shape")
    return mx.contiguous(result.astype(mx.int32))


def rope(
    x: mx.array | np.ndarray,
    positions: mx.array | np.ndarray,
    *,
    base: float = 1_000_000.0,
    rotary_dim: int | None = None,
    sections: Sequence[int] | None = None,
    table_len: int | None = None,
    sequence_axis: int = -2,
    frequency_dim: int | None = None,
    active_pairs: int | None = None,
) -> mx.array:
    """Apply rotate-half RoPE, including partial RoPE and three-axis MRoPE."""

    source = _floating(x)
    if source.ndim < 2:
        raise ValueError("RoPE input must have at least two dimensions")
    axis = int(sequence_axis) % source.ndim
    if axis == source.ndim - 1:
        raise ValueError("RoPE sequence axis cannot be the final feature axis")
    moved = axis != source.ndim - 2
    canonical = mx.moveaxis(source, axis, -2) if moved else source
    canonical = mx.contiguous(canonical)
    shape = tuple(int(value) for value in canonical.shape)
    tokens, dimension = shape[-2:]
    rotary = dimension if rotary_dim is None else int(rotary_dim)
    if rotary <= 0 or rotary > dimension or rotary % 2:
        raise ValueError("rotary_dim must be positive, even, and no larger than the head dimension")

    position_ids = _position_array(positions)
    if int(position_ids.shape[-1]) != tokens:
        raise ValueError(
            f"RoPE position length {position_ids.shape[-1]} != sequence length {tokens}"
        )
    position_axes = 1 if position_ids.ndim == 1 else int(position_ids.shape[0])
    section_values = (0, 0, 0)
    if sections is not None:
        if len(sections) != 3:
            raise ValueError("MRoPE sections must contain exactly three entries")
        section_values = tuple(int(value) for value in sections)
        if any(value < 0 for value in section_values) or sum(section_values) != rotary // 2:
            raise ValueError("MRoPE sections must be nonnegative and sum to rotary_dim / 2")

    if tokens == 0 or int(canonical.size) == 0:
        output = mx.zeros(shape, dtype=canonical.dtype)
        return mx.moveaxis(output, -2, axis) if moved else output
    if table_len is None:
        maximum = mx.max(position_ids)
        mx.eval(maximum)
        table_len = max(16, int(maximum.item()) + 1)
    cosine, sine = rope_tables(
        float(base),
        rotary,
        int(table_len),
        frequency_dim=frequency_dim,
        active_pairs=active_pairs,
    )
    size = int(canonical.size)
    output = _ROPE_KERNEL(
        inputs=[canonical, position_ids, cosine, sine],
        template=[
            ("T", canonical.dtype),
            ("SIZE", size),
            ("TOKENS", tokens),
            ("DIM", dimension),
            ("ROTARY_DIM", rotary),
            ("POS_AXES", position_axes),
            ("TABLE_LEN", int(table_len)),
            ("S0", section_values[0]),
            ("S1", section_values[1]),
            ("S2", section_values[2]),
        ],
        grid=(size, 1, 1),
        threadgroup=_threadgroup_size(size),
        output_shapes=[shape],
        output_dtypes=[canonical.dtype],
    )[0]
    return mx.moveaxis(output, -2, axis) if moved else output


__all__ = [
    "gelu_mul",
    "hadamard_mul",
    "l2_norm",
    "residual_add",
    "residual_rms_norm",
    "rms_norm",
    "rope",
    "rope_tables",
    "silu_mul",
]
