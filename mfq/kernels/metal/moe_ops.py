"""Router and post-processing Metal kernels for routed MoE layers."""

from __future__ import annotations

import math

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_TOPK_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float transformed[EXPERTS];
    threadgroup float partial[256];
    uint row_offset = row * uint(EXPERTS);

    float local_max = -INFINITY;
    if (MODE == 0) {
        for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
            float raw = float(logits[row_offset + expert]);
            raw = isnan(raw) ? -FLT_MAX : raw;
            local_max = max(local_max, raw);
        }
        partial[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] = max(partial[tid], partial[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    float maximum = partial[0];

    float local_sum = 0.0f;
    for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
        float raw = float(logits[row_offset + expert]);
        raw = isnan(raw) ? -FLT_MAX : raw;
        float value;
        if (MODE == 0) {
            value = exp(raw - maximum);
            local_sum += value;
        } else if (MODE == 1) {
            value = 1.0f / (1.0f + exp(-raw));
        } else if (MODE == 2) {
            float softplus = raw > 20.0f ? raw : log1p(exp(raw));
            value = sqrt(softplus);
        } else {
            value = raw;
        }
        transformed[expert] = value;
    }
    if (MODE == 0) {
        partial[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float denominator = partial[0];
        for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
            transformed[expert] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float selected_weights[TOP_K];
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float best_score = -INFINITY;
            uint best_expert = uint(EXPERTS);
            for (uint expert = 0u; expert < uint(EXPERTS); ++expert) {
                float weight = transformed[expert];
                float score = (
                    HAS_AVAILABLE == 0 || available[expert]
                )
                    ? weight + (HAS_BIAS != 0 ? bias[expert] : 0.0f)
                    : -INFINITY;
                if (
                    score > best_score
                    || (score == best_score && expert < best_expert)
                ) {
                    best_score = score;
                    best_expert = expert;
                }
            }
            uint output_index = row * uint(TOP_K) + rank;
            ids[output_index] = int(best_expert);
            selected_weights[rank] = transformed[best_expert];
            transformed[best_expert] = -INFINITY;
        }

        float denominator = 1.0f;
        if (MODE == 3) {
            float selected_max = -INFINITY;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                selected_max = max(selected_max, selected_weights[rank]);
            }
            denominator = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                selected_weights[rank] = exp(
                    selected_weights[rank] - selected_max
                );
                denominator += selected_weights[rank];
            }
        } else if (NORMALIZE != 0) {
            denominator = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                denominator += selected_weights[rank];
            }
            denominator = max(denominator, params[0]);
        }
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float value = selected_weights[rank];
            if (MODE == 3 || NORMALIZE != 0) {
                value /= denominator;
            }
            weights[row * uint(TOP_K) + rank] = value * params[1];
        }
    }
"""


_SQRTSOFTPLUS_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }
    float value = 0.0f;
    if (lane < uint(TOP_K)) {
        uint expert = uint(ids[row * uint(TOP_K) + lane]);
        float raw = float(logits[row * uint(EXPERTS) + expert]);
        float softplus = raw > 20.0f ? raw : log1p(exp(raw));
        value = sqrt(softplus);
    }
    float denominator = max(simd_sum(value), params[0]);
    if (lane < uint(TOP_K)) {
        weights[row * uint(TOP_K) + lane] = value / denominator * params[1];
    }
"""


_WEIGHTED_REDUCE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    uint column = index - token * uint(WIDTH);
    float value = 0.0f;
    for (uint route = 0u; route < uint(ROUTES); ++route) {
        value += float(pair_output[
            (token * uint(ROUTES) + route) * uint(WIDTH) + column
        ]) * weights[token * uint(ROUTES) + route];
    }
    output[index] = T(value);
"""


_GLU_SPLIT_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(ROWS * WIDTH)) {
        return;
    }
    uint row = index / uint(WIDTH);
    uint column = index - row * uint(WIDTH);
    uint offset = row * uint(WIDTH * 2);
    float gate = float(gate_up[offset + column]);
    float up = float(gate_up[offset + uint(WIDTH) + column]);
    float activated;
    if (GEGLU != 0) {
        constexpr float gelu_scale = 0.7978845608028654f;
        float inner = gelu_scale * (gate + 0.044715f * gate * gate * gate);
        activated = 0.5f * gate * (1.0f + tanh(inner));
    } else {
        activated = gate / (1.0f + exp(-gate));
    }
    output[index] = T(activated * up);
"""


_SHARED_GATE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    float gate = 1.0f / (1.0f + exp(-gate_logits[token]));
    output[index] = T(float(routed[index]) + gate * float(shared[index]));
"""


_REDUCE_SHARED_GATE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    uint column = index - token * uint(WIDTH);
    float value = 0.0f;
    for (uint route = 0u; route < uint(ROUTES); ++route) {
        value += float(pair_output[
            (token * uint(ROUTES) + route) * uint(WIDTH) + column
        ]) * weights[token * uint(ROUTES) + route];
    }
    T routed_value = T(value);
    float gate = 1.0f / (1.0f + exp(-gate_logits[token]));
    output[index] = T(float(routed_value) + gate * float(shared[index]));
"""


_EXPERT_SCALE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    output[index] = weights[index] * scales[uint(ids[index])];
"""


_TOPK_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_topk",
    input_names=["logits", "bias", "available", "params"],
    output_names=["ids", "weights"],
    source=_TOPK_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SQRTSOFTPLUS_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_sqrtsoftplus_weights",
    input_names=["logits", "ids", "params"],
    output_names=["weights"],
    source=_SQRTSOFTPLUS_SOURCE,
    compile_options={"math_mode": "fast"},
)

_WEIGHTED_REDUCE_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_weighted_reduce",
    input_names=["pair_output", "weights"],
    output_names=["output"],
    source=_WEIGHTED_REDUCE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GLU_SPLIT_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_glu_split",
    input_names=["gate_up"],
    output_names=["output"],
    source=_GLU_SPLIT_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SHARED_GATE_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_shared_gate",
    input_names=["routed", "shared", "gate_logits"],
    output_names=["output"],
    source=_SHARED_GATE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_REDUCE_SHARED_GATE_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_reduce_shared_gate",
    input_names=["pair_output", "weights", "shared", "gate_logits"],
    output_names=["output"],
    source=_REDUCE_SHARED_GATE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_EXPERT_SCALE_KERNEL = mx.fast.metal_kernel(
    name="mfq_moe_expert_scale",
    input_names=["weights", "ids", "scales"],
    output_names=["output"],
    source=_EXPERT_SCALE_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def _float32(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(mx.float32))


def _ids(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(mx.int32))


def moe_topk(
    logits: mx.array | np.ndarray,
    top_k: int,
    *,
    use_sigmoid: bool = False,
    use_sqrt_softplus: bool = False,
    normalize: bool = False,
    delayed_softmax: bool = False,
    bias: mx.array | np.ndarray | None = None,
    available: mx.array | np.ndarray | None = None,
    norm_floor: float = 1e-20,
    scale: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Apply the CUDA-compatible router transform and select top-k experts."""

    values = _floating(logits)
    if values.ndim < 2:
        raise ValueError("MoE router logits must end in an expert dimension")
    experts = int(values.shape[-1])
    rows = int(values.size) // experts
    selected = int(top_k)
    if not 1 <= selected <= min(16, experts):
        raise ValueError("top_k must be in [1,min(16,experts)]")
    if experts > 4096:
        raise ValueError("Metal MoE top-k currently supports at most 4096 experts")
    modes = int(use_sigmoid) + int(use_sqrt_softplus) + int(delayed_softmax)
    if modes > 1:
        raise ValueError("sigmoid, sqrt-softplus, and delayed softmax are exclusive")
    if delayed_softmax and normalize:
        raise ValueError("normalize and delayed_softmax are mutually exclusive")
    if not math.isfinite(norm_floor) or norm_floor < 0.0:
        raise ValueError("norm_floor must be finite and non-negative")
    if not math.isfinite(scale):
        raise ValueError("router scale must be finite")
    mode = 1 if use_sigmoid else (2 if use_sqrt_softplus else (3 if delayed_softmax else 0))
    if bias is None:
        bias_values = mx.zeros((experts,), dtype=mx.float32)
        has_bias = False
    else:
        bias_values = _float32(bias)
        if tuple(bias_values.shape) != (experts,):
            raise ValueError(f"router bias must have shape ({experts},)")
        has_bias = True
    if available is None:
        available_values = mx.ones((experts,), dtype=mx.bool_)
        has_available = False
    else:
        available_values = (
            available if isinstance(available, mx.array) else mx.array(available)
        ).astype(mx.bool_)
        available_values = mx.contiguous(available_values.reshape((-1,)))
        if tuple(available_values.shape) != (experts,):
            raise ValueError(f"router availability must have shape ({experts},)")
        has_available = True
    params = mx.array([float(norm_floor), float(scale)], dtype=mx.float32)
    outputs = _TOPK_KERNEL(
        inputs=[
            values.reshape((rows, experts)),
            bias_values,
            available_values,
            params,
        ],
        template=[
            ("T", values.dtype),
            ("ROWS", rows),
            ("EXPERTS", experts),
            ("TOP_K", selected),
            ("MODE", mode),
            ("NORMALIZE", int(normalize)),
            ("HAS_BIAS", int(has_bias)),
            ("HAS_AVAILABLE", int(has_available)),
        ],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, selected), (rows, selected)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return outputs[0], outputs[1]


def sqrtsoftplus_weights(
    logits: mx.array | np.ndarray,
    ids: mx.array | np.ndarray,
    *,
    norm_floor: float = 1e-20,
    scale: float = 1.0,
) -> mx.array:
    """Gather and normalize DeepSeek-style sqrt-softplus router weights."""

    values = _floating(logits)
    selected = _ids(ids)
    if values.ndim < 2 or selected.ndim != 2:
        raise ValueError("logits and ids must be [rows,experts] and [rows,top_k]")
    experts = int(values.shape[-1])
    rows, top_k = map(int, selected.shape)
    if int(values.size) != rows * experts or not 1 <= top_k <= 16:
        raise ValueError("sqrtsoftplus router shapes are incompatible")
    params = mx.array([float(norm_floor), float(scale)], dtype=mx.float32)
    return _SQRTSOFTPLUS_KERNEL(
        inputs=[values.reshape((rows, experts)), selected, params],
        template=[
            ("T", values.dtype),
            ("ROWS", rows),
            ("EXPERTS", experts),
            ("TOP_K", top_k),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[selected.shape],
        output_dtypes=[mx.float32],
    )[0]


def weighted_reduce(
    pair_output: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
) -> mx.array:
    """Reduce ``[tokens,routes,width]`` with FP32 accumulation."""

    pairs = _floating(pair_output)
    route_weights = _float32(weights)
    if pairs.ndim != 3:
        raise ValueError("pair_output must have shape [tokens,routes,width]")
    tokens, routes, width = map(int, pairs.shape)
    if tuple(route_weights.shape) != (tokens, routes):
        raise ValueError("route weights must have shape [tokens,routes]")
    size = tokens * width
    return _WEIGHTED_REDUCE_KERNEL(
        inputs=[pairs, route_weights],
        template=[
            ("T", pairs.dtype),
            ("TOKENS", tokens),
            ("ROUTES", routes),
            ("WIDTH", width),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[pairs.dtype],
    )[0]


def _glu_split(
    gate_up: mx.array | np.ndarray,
    *,
    geglu: bool,
) -> mx.array:
    values = _floating(gate_up)
    if values.ndim < 2 or int(values.shape[-1]) % 2:
        raise ValueError("gate_up must end in an even 2*width dimension")
    width = int(values.shape[-1]) // 2
    rows = int(values.size) // (2 * width)
    shape = (*map(int, values.shape[:-1]), width)
    size = rows * width
    return _GLU_SPLIT_KERNEL(
        inputs=[values],
        template=[
            ("T", values.dtype),
            ("ROWS", rows),
            ("WIDTH", width),
            ("GEGLU", int(geglu)),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[shape],
        output_dtypes=[values.dtype],
    )[0]


def swiglu_split(gate_up: mx.array | np.ndarray) -> mx.array:
    return _glu_split(gate_up, geglu=False)


def geglu_split(gate_up: mx.array | np.ndarray) -> mx.array:
    return _glu_split(gate_up, geglu=True)


def add_shared_gate(
    routed: mx.array | np.ndarray,
    shared: mx.array | np.ndarray,
    gate_logits: mx.array | np.ndarray,
) -> mx.array:
    routed_values = _floating(routed)
    shared_values = _floating(shared).astype(routed_values.dtype)
    gates = _float32(gate_logits)
    if routed_values.ndim != 2 or tuple(shared_values.shape) != tuple(routed_values.shape):
        raise ValueError("routed and shared must have matching [tokens,width] shapes")
    tokens, width = map(int, routed_values.shape)
    if int(gates.size) != tokens:
        raise ValueError("shared gate logits must contain one value per token")
    size = tokens * width
    return _SHARED_GATE_KERNEL(
        inputs=[routed_values, shared_values, gates.reshape((tokens,))],
        template=[("T", routed_values.dtype), ("TOKENS", tokens), ("WIDTH", width)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[routed_values.shape],
        output_dtypes=[routed_values.dtype],
    )[0]


def weighted_reduce_shared_gate(
    pair_output: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
    shared: mx.array | np.ndarray,
    gate_logits: mx.array | np.ndarray,
) -> mx.array:
    pairs = _floating(pair_output)
    route_weights = _float32(weights)
    shared_values = _floating(shared).astype(pairs.dtype)
    gates = _float32(gate_logits)
    if pairs.ndim != 3:
        raise ValueError("pair_output must have shape [tokens,routes,width]")
    tokens, routes, width = map(int, pairs.shape)
    if tuple(route_weights.shape) != (tokens, routes):
        raise ValueError("route weights must have shape [tokens,routes]")
    if tuple(shared_values.shape) != (tokens, width) or int(gates.size) != tokens:
        raise ValueError("shared output or gate shape is incompatible")
    size = tokens * width
    return _REDUCE_SHARED_GATE_KERNEL(
        inputs=[
            pairs,
            route_weights,
            shared_values,
            gates.reshape((tokens,)),
        ],
        template=[
            ("T", pairs.dtype),
            ("TOKENS", tokens),
            ("ROUTES", routes),
            ("WIDTH", width),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[pairs.dtype],
    )[0]


def apply_expert_scale(
    weights: mx.array | np.ndarray,
    ids: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
) -> mx.array:
    values = _float32(weights)
    selected = _ids(ids)
    expert_scales = _float32(scales)
    if tuple(values.shape) != tuple(selected.shape) or expert_scales.ndim != 1:
        raise ValueError("weights/ids shapes or expert scales are invalid")
    size = int(values.size)
    return _EXPERT_SCALE_KERNEL(
        inputs=[values, selected, expert_scales],
        template=[("SIZE", size)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]


__all__ = [
    "add_shared_gate",
    "apply_expert_scale",
    "geglu_split",
    "moe_topk",
    "sqrtsoftplus_weights",
    "swiglu_split",
    "weighted_reduce",
    "weighted_reduce_shared_gate",
]
