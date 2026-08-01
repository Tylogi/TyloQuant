"""Q8_1 activation quantization for Apple silicon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_QUANTIZE_RECONSTRUCT_SOURCE = r"""
    constexpr uint GROUP_SIZE = 32u;
    uint lane = thread_index_in_simdgroup;
    uint group_index = threadgroup_position_in_grid.x;
    uint row = group_index / uint(GROUPS);
    uint group = group_index - row * uint(GROUPS);
    uint column = group * GROUP_SIZE + lane;
    float value = column < uint(K)
        ? float(x[row * uint(K) + column])
        : 0.0f;

    float amax = simd_max(abs(value));
    float scale = amax / 127.0f;
    float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    float scaled = value * inverse;
    int code = int(
        scaled >= 0.0f
        ? floor(scaled + 0.5f)
        : ceil(scaled - 0.5f)
    );
    code = clamp(code, -127, 127);
    q[group_index * GROUP_SIZE + lane] = char(code);

    float code_sum = simd_sum(float(code));
    half stored_scale = half(scale);
    if (lane == 0u) {
        d[group_index] = stored_scale;
        s[group_index] = half(code_sum * scale);
    }
    if (column < uint(K)) {
        reconstructed[row * uint(K) + column] =
            half(float(code) * float(stored_scale));
    }
"""

_QUANTIZE_RECONSTRUCT_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint8_one_quantize_reconstruct",
    input_names=["x"],
    output_names=["q", "d", "s", "reconstructed"],
    source=_QUANTIZE_RECONSTRUCT_SOURCE,
    compile_options={"math_mode": "fast"},
)


@dataclass(frozen=True)
class MetalNint8OneBlocks:
    """Q8_1 activation blocks resident in Metal memory."""

    q: mx.array
    d: mx.array
    s: mx.array
    reconstructed: mx.array


def nint8_one_quantize_reconstruct(
    values: mx.array | np.ndarray,
) -> MetalNint8OneBlocks:
    """Quantize FP16 activations to Q8_1 and reconstruct them in one dispatch."""

    source = values if isinstance(values, mx.array) else mx.array(values)
    if source.ndim < 1 or int(source.size) == 0 or int(source.shape[-1]) == 0:
        raise ValueError("NINT8-1 input must be non-empty")
    if source.dtype != mx.float16:
        raise ValueError("NINT8-1 Metal input must be FP16")
    source = mx.contiguous(source)
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    width = int(source.shape[-1])
    groups = (width + 31) // 32
    q, d, s, reconstructed = _QUANTIZE_RECONSTRUCT_KERNEL(
        inputs=[source.reshape((rows, width))],
        template=[
            ("M", rows),
            ("K", width),
            ("GROUPS", groups),
        ],
        grid=(rows * groups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (*prefix, groups, 32),
            (*prefix, groups),
            (*prefix, groups),
            (*prefix, width),
        ],
        output_dtypes=[
            mx.int8,
            mx.float16,
            mx.float16,
            mx.float16,
        ],
    )
    return MetalNint8OneBlocks(
        q=q,
        d=d,
        s=s,
        reconstructed=reconstructed,
    )


__all__ = [
    "MetalNint8OneBlocks",
    "nint8_one_quantize_reconstruct",
]
