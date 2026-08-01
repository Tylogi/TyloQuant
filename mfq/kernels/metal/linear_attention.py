"""Metal kernels used by gated linear-attention models.

The Gated DeltaNet kernel mirrors the CUDA warp-column implementation: one
SIMD group owns one output column of the recurrent state and keeps that column
in registers across the token loop.  The fused linear-convolution kernel
combines causal depthwise convolution, SiLU, Q/K L2 normalization, and cache
state production in one dispatch.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_GDN_SOURCE = r"""
    constexpr uint SIMD_WIDTH = 32u;
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint ROWS = (uint(D) + SIMD_WIDTH - 1u) / SIMD_WIDTH;
    constexpr uint COLUMN_TILES = (uint(D) + SIMD_GROUPS - 1u) / SIMD_GROUPS;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint bh = workgroup / COLUMN_TILES;
    uint column = (workgroup - bh * COLUMN_TILES) * SIMD_GROUPS + simd_group;
    if (bh >= uint(B * HV) || column >= uint(D)) {
        return;
    }

    uint batch = bh / uint(HV);
    uint value_head = bh - batch * uint(HV);
    uint query_head = TILED_HEADS != 0
        ? value_head % uint(HQ)
        : value_head / uint(HV / HQ);
    uint state_offset = bh * uint(D * D);
    uint query_sequence = (batch * uint(HQ) + query_head) * uint(TOKENS);
    uint value_sequence = bh * uint(TOKENS);

    float state_values[ROWS];
    for (uint row = 0u; row < ROWS; ++row) {
        uint state_row = row * SIMD_WIDTH + lane;
        uint state_index = TRANSPOSED_STATE != 0
            ? column * uint(D) + state_row
            : state_row * uint(D) + column;
        state_values[row] = state_row < uint(D)
            ? state_in[state_offset + state_index]
            : 0.0f;
    }

    const float scale = 1.0f / sqrt(float(D));
    for (uint token = 0u; token < uint(TOKENS); ++token) {
        uint query_offset = (query_sequence + token) * uint(D);
        uint value_offset = (value_sequence + token) * uint(D);
        float key_values[ROWS];
        float query_values[ROWS];
        for (uint row = 0u; row < ROWS; ++row) {
            uint state_row = row * SIMD_WIDTH + lane;
            key_values[row] = state_row < uint(D) ? k[query_offset + state_row] : 0.0f;
            query_values[row] = state_row < uint(D) ? q[query_offset + state_row] : 0.0f;
        }

        float projected_key = 0.0f;
        if (KDA != 0) {
            for (uint row = 0u; row < ROWS; ++row) {
                uint state_row = row * SIMD_WIDTH + lane;
                float decay = state_row < uint(D)
                    ? exp(g[value_offset + state_row])
                    : 0.0f;
                projected_key += decay * state_values[row] * key_values[row];
            }
        } else {
            for (uint row = 0u; row < ROWS; ++row) {
                projected_key += state_values[row] * key_values[row];
            }
        }
        projected_key = simd_sum(projected_key);

        float beta_value = beta[value_sequence + token];
        float delta;
        if (KDA != 0) {
            delta = (v[value_offset + column] - projected_key) * beta_value;
        } else {
            float decay = exp(g[value_sequence + token]);
            delta = (v[value_offset + column] - decay * projected_key) * beta_value;
        }

        float result = 0.0f;
        if (KDA != 0) {
            for (uint row = 0u; row < ROWS; ++row) {
                uint state_row = row * SIMD_WIDTH + lane;
                float decay = state_row < uint(D)
                    ? exp(g[value_offset + state_row])
                    : 0.0f;
                state_values[row] =
                    decay * state_values[row] + key_values[row] * delta;
                result += state_values[row] * query_values[row];
            }
        } else {
            float decay = exp(g[value_sequence + token]);
            for (uint row = 0u; row < ROWS; ++row) {
                state_values[row] =
                    decay * state_values[row] + key_values[row] * delta;
                result += state_values[row] * query_values[row];
            }
        }
        result = simd_sum(result);
        if (lane == 0u) {
            out[value_offset + column] = result * scale;
        }
    }

    for (uint row = 0u; row < ROWS; ++row) {
        uint state_row = row * SIMD_WIDTH + lane;
        if (state_row < uint(D)) {
            uint state_index = TRANSPOSED_STATE != 0
                ? column * uint(D) + state_row
                : state_row * uint(D) + column;
            state_out[state_offset + state_index] = state_values[row];
        }
    }
"""


_SSM_CONV_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(B * TOKENS * C)) {
        return;
    }
    uint channel = index % uint(C);
    uint row = index / uint(C);
    uint token = row % uint(TOKENS);
    uint batch = row / uint(TOKENS);
    uint input_offset = (batch * uint(TOKENS + K - 1) + token) * uint(C) + channel;
    float value = HAS_BIAS != 0 ? bias[channel] : 0.0f;
    for (uint tap = 0u; tap < uint(K); ++tap) {
        value += float(x[input_offset + tap * uint(C)])
            * weight[channel * uint(K) + tap];
    }
    out[index] = value / (1.0f + exp(-value));
"""


_LINEAR_CONV_QKV_SOURCE = r"""
    constexpr uint QK_TASKS = uint(B * TOKENS * 2 * NK);
    constexpr uint V_GROUPS = (uint(NV * DV) + 31u) / 32u;
    constexpr uint V_TASKS = uint(B * TOKENS) * V_GROUPS;
    constexpr uint STATE_SIZE = uint(B * (K - 1) * C);
    constexpr uint STATE_TASKS = (STATE_SIZE + 31u) / 32u;

    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;

    if (task < QK_TASKS) {
        uint head = task % uint(NK);
        uint row = task / uint(NK);
        uint which = row & 1u;
        uint batch_token = row >> 1u;
        uint token = batch_token % uint(TOKENS);
        uint batch = batch_token / uint(TOKENS);
        uint channel_base = (which * uint(NK) + head) * uint(DK);
        float square_sum = 0.0f;
        float values[(uint(DK) + 31u) / 32u];

        uint local = 0u;
        for (uint dimension = lane; dimension < uint(DK); dimension += 32u) {
            uint channel = channel_base + dimension;
            float value = HAS_BIAS != 0 ? bias[channel] : 0.0f;
            for (uint tap = 0u; tap < uint(K); ++tap) {
                int source_token = int(token) + int(tap) - int(K - 1);
                float source;
                if (source_token < 0) {
                    uint state_row = uint(source_token + int(K - 1));
                    source = state_in[
                        (batch * uint(K - 1) + state_row) * uint(C) + channel
                    ];
                } else {
                    source = float(qk[
                        (batch * uint(TOKENS) + uint(source_token))
                            * uint(QKC) + channel
                    ]);
                }
                value += source * weight[channel * uint(K) + tap];
            }
            value = value / (1.0f + exp(-value));
            values[local++] = value;
            square_sum += value * value;
        }
        square_sum = simd_sum(square_sum);
        float inverse = 1.0f / max(sqrt(square_sum), params[0]);
        local = 0u;
        for (uint dimension = lane; dimension < uint(DK); dimension += 32u) {
            uint output_index =
                (((batch * uint(NK) + head) * uint(TOKENS) + token)
                    * uint(DK) + dimension);
            if (which == 0u) {
                q_out[output_index] = values[local++] * inverse;
            } else {
                k_out[output_index] = values[local++] * inverse;
            }
        }
        return;
    }

    task -= QK_TASKS;
    if (task < V_TASKS) {
        uint value_group = task % V_GROUPS;
        uint batch_token = task / V_GROUPS;
        uint token = batch_token % uint(TOKENS);
        uint batch = batch_token / uint(TOKENS);
        uint value_index = value_group * 32u + lane;
        if (value_index < uint(NV * DV)) {
            uint channel = uint(QKC) + value_index;
            float value = HAS_BIAS != 0 ? bias[channel] : 0.0f;
            for (uint tap = 0u; tap < uint(K); ++tap) {
                int source_token = int(token) + int(tap) - int(K - 1);
                float source;
                if (source_token < 0) {
                    uint state_row = uint(source_token + int(K - 1));
                    source = state_in[
                        (batch * uint(K - 1) + state_row) * uint(C) + channel
                    ];
                } else {
                    source = float(v_in[
                        (batch * uint(TOKENS) + uint(source_token))
                            * uint(NV * DV) + value_index
                    ]);
                }
                value += source * weight[channel * uint(K) + tap];
            }
            value = value / (1.0f + exp(-value));
            uint head = value_index / uint(DV);
            uint dimension = value_index - head * uint(DV);
            v_out[
                (((batch * uint(NV) + head) * uint(TOKENS) + token)
                    * uint(DV) + dimension)
            ] = value;
        }
        return;
    }

    task -= V_TASKS;
    if (task < STATE_TASKS) {
        uint index = task * 32u + lane;
        if (index < STATE_SIZE) {
            uint channel = index % uint(C);
            uint row = index / uint(C);
            uint state_row = row % uint(K - 1);
            uint batch = row / uint(K - 1);
            uint combined = uint(TOKENS) + state_row;
            if (combined < uint(K - 1)) {
                state_out[index] = state_in[
                    (batch * uint(K - 1) + combined) * uint(C) + channel
                ];
            } else {
                uint source_token = combined - uint(K - 1);
                if (channel < uint(QKC)) {
                    state_out[index] = float(qk[
                        (batch * uint(TOKENS) + source_token)
                            * uint(QKC) + channel
                    ]);
                } else {
                    state_out[index] = float(v_in[
                        (batch * uint(TOKENS) + source_token)
                            * uint(NV * DV) + channel - uint(QKC)
                    ]);
                }
            }
        }
    }
"""


_GDN_KERNEL = mx.fast.metal_kernel(
    name="mfq_gated_delta_net",
    input_names=["q", "k", "v", "g", "beta", "state_in"],
    output_names=["out", "state_out"],
    source=_GDN_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SSM_CONV_KERNEL = mx.fast.metal_kernel(
    name="mfq_ssm_conv_silu",
    input_names=["x", "weight", "bias"],
    output_names=["out"],
    source=_SSM_CONV_SOURCE,
    compile_options={"math_mode": "fast"},
)

_LINEAR_CONV_QKV_KERNEL = mx.fast.metal_kernel(
    name="mfq_linear_conv_qkv",
    input_names=["state_in", "qk", "v_in", "weight", "bias", "params"],
    output_names=["q_out", "k_out", "v_out", "state_out"],
    source=_LINEAR_CONV_QKV_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _float32(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(mx.float32))


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def gated_delta_net(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    v: mx.array | np.ndarray,
    g: mx.array | np.ndarray,
    beta: mx.array | np.ndarray,
    state: mx.array | np.ndarray | None = None,
    *,
    transposed_state: bool = False,
    tiled_heads: bool = False,
) -> tuple[mx.array, mx.array]:
    """Run the Gated DeltaNet recurrence in a column-sharded Metal kernel.

    ``q`` and ``k`` use ``[B,Hq,T,D]`` while ``v`` uses ``[B,Hv,T,D]``.
    ``Hv`` must be divisible by ``Hq``.  Grouped head mapping is contiguous by
    default; ``tiled_heads=True`` selects the GGUF tiled mapping.
    """

    query = _float32(q)
    key = _float32(k)
    value = _float32(v)
    gate = _float32(g)
    beta_values = _float32(beta)
    if query.ndim != 4:
        raise ValueError("GDN q must have shape [B,Hq,T,D]")
    batch, query_heads, tokens, dimension = map(int, query.shape)
    if dimension not in (32, 64, 128):
        raise ValueError(f"GDN Metal kernel supports D in {{32,64,128}}, got {dimension}")
    if tuple(key.shape) != tuple(query.shape):
        raise ValueError(f"GDN k shape {key.shape} must match q shape {query.shape}")
    if value.ndim != 4:
        raise ValueError("GDN v must have shape [B,Hv,T,D]")
    value_heads = int(value.shape[1])
    if tuple(value.shape) != (batch, value_heads, tokens, dimension):
        raise ValueError(
            f"GDN v must have shape ({batch},Hv,{tokens},{dimension}), got {value.shape}"
        )
    if query_heads <= 0 or value_heads % query_heads != 0:
        raise ValueError("GDN value-head count must be divisible by query-head count")
    kda = gate.ndim == 4
    expected_gate = (batch, value_heads, tokens, dimension) if kda else (batch, value_heads, tokens)
    if tuple(gate.shape) != expected_gate:
        raise ValueError(f"GDN gate must have shape {expected_gate}, got {gate.shape}")
    if tuple(beta_values.shape) != (batch, value_heads, tokens):
        raise ValueError(
            f"GDN beta must have shape {(batch, value_heads, tokens)}, got {beta_values.shape}"
        )
    state_shape = (batch, value_heads, dimension, dimension)
    state_values = mx.zeros(state_shape, dtype=mx.float32) if state is None else _float32(state)
    if tuple(state_values.shape) != state_shape:
        raise ValueError(f"GDN state must have shape {state_shape}, got {state_values.shape}")
    if tokens == 0:
        return mx.zeros(value.shape, dtype=mx.float32), mx.array(state_values)

    column_tiles = (dimension + 3) // 4
    workgroups = batch * value_heads * column_tiles
    outputs = _GDN_KERNEL(
        inputs=[query, key, value, gate, beta_values, state_values],
        template=[
            ("B", batch),
            ("HQ", query_heads),
            ("HV", value_heads),
            ("TOKENS", tokens),
            ("D", dimension),
            ("KDA", int(kda)),
            ("TRANSPOSED_STATE", int(transposed_state)),
            ("TILED_HEADS", int(tiled_heads)),
        ],
        grid=(workgroups * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[value.shape, state_shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs[0], outputs[1]


def _conv_weight(
    weight: mx.array | np.ndarray,
    channels: int,
) -> tuple[mx.array, int]:
    source = _float32(weight)
    if source.ndim == 3:
        if int(source.shape[0]) != channels or int(source.shape[1]) != 1:
            raise ValueError(f"SSM [C,1,K] weight must start with ({channels},1)")
        kernel = int(source.shape[2])
        source = source.reshape((channels, kernel))
    elif source.ndim == 2 and int(source.shape[0]) == channels:
        kernel = int(source.shape[1])
    elif source.ndim == 2 and int(source.shape[1]) == channels:
        kernel = int(source.shape[0])
        source = mx.transpose(source)
    else:
        raise ValueError(f"SSM weight must have shape [C,1,K], [C,K], or [K,C] for C={channels}")
    if kernel <= 0:
        raise ValueError("SSM convolution kernel width must be positive")
    return mx.contiguous(source), kernel


def _conv_bias(
    bias: mx.array | np.ndarray | None,
    channels: int,
) -> tuple[mx.array, bool]:
    if bias is None:
        return mx.zeros((channels,), dtype=mx.float32), False
    result = _float32(bias)
    if tuple(result.shape) != (channels,):
        raise ValueError(f"SSM bias must have shape ({channels},), got {result.shape}")
    return result, True


def ssm_conv_silu(
    conv_input: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
    n_tokens: int,
    bias: mx.array | np.ndarray | None = None,
) -> mx.array:
    """Fused causal depthwise convolution and SiLU.

    ``conv_input`` has shape ``[B,K-1+T,C]``. Weight layouts ``[C,1,K]``,
    ``[C,K]``, and ``[K,C]`` are accepted.
    """

    source = _floating(conv_input)
    if source.ndim != 3:
        raise ValueError("SSM convolution input must have shape [B,K-1+T,C]")
    batch, length, channels = map(int, source.shape)
    tokens = int(n_tokens)
    packed_weight, kernel = _conv_weight(weight, channels)
    bias_values, has_bias = _conv_bias(bias, channels)
    if tokens <= 0 or length != tokens + kernel - 1:
        raise ValueError(f"SSM input length must be T+K-1={tokens + kernel - 1}, got {length}")
    size = batch * tokens * channels
    return _SSM_CONV_KERNEL(
        inputs=[source, packed_weight, bias_values],
        template=[
            ("T", source.dtype),
            ("B", batch),
            ("TOKENS", tokens),
            ("C", channels),
            ("K", kernel),
            ("HAS_BIAS", int(has_bias)),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(batch, tokens, channels)],
        output_dtypes=[mx.float32],
    )[0]


def linear_conv_qkv(
    state: mx.array | np.ndarray,
    qk: mx.array | np.ndarray,
    v: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    bias: mx.array | np.ndarray | None = None,
    eps: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused model-specific SSM convolution, Q/K normalization, and cache update."""

    qk_values = _floating(qk)
    v_values = _floating(v)
    state_values = _float32(state)
    if qk_values.ndim != 3 or v_values.ndim != 3 or state_values.ndim != 3:
        raise ValueError("linear_conv_qkv expects rank-3 state, qk, and v arrays")
    batch, tokens, qk_channels = map(int, qk_values.shape)
    nk = int(num_key_heads)
    nv = int(num_value_heads)
    dk = int(key_head_dim)
    dv = int(value_head_dim)
    if min(batch, tokens, nk, nv, dk, dv) <= 0:
        raise ValueError("linear_conv_qkv dimensions must be positive")
    if nv % nk != 0:
        raise ValueError("linear_conv_qkv value heads must be divisible by key heads")
    expected_qk = 2 * nk * dk
    expected_v = nv * dv
    if qk_channels != expected_qk:
        raise ValueError(f"qk width must be {expected_qk}, got {qk_channels}")
    if tuple(v_values.shape) != (batch, tokens, expected_v):
        raise ValueError(f"v must have shape {(batch, tokens, expected_v)}, got {v_values.shape}")
    if v_values.dtype != qk_values.dtype:
        dtype = mx.float32 if mx.float32 in (qk_values.dtype, v_values.dtype) else mx.float16
        qk_values = mx.contiguous(qk_values.astype(dtype))
        v_values = mx.contiguous(v_values.astype(dtype))
    channels = expected_qk + expected_v
    if int(state_values.shape[0]) != batch or int(state_values.shape[2]) != channels:
        raise ValueError(f"state must have shape [B,K-1,{channels}], got {state_values.shape}")
    kernel = int(state_values.shape[1]) + 1
    if kernel <= 1:
        raise ValueError("linear_conv_qkv requires a convolution kernel width of at least 2")
    packed_weight, weight_kernel = _conv_weight(weight, channels)
    if weight_kernel != kernel:
        raise ValueError(f"state implies K={kernel}, but weight uses K={weight_kernel}")
    bias_values, has_bias = _conv_bias(bias, channels)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("linear_conv_qkv eps must be finite and positive")

    qk_tasks = batch * tokens * 2 * nk
    v_groups = (expected_v + 31) // 32
    v_tasks = batch * tokens * v_groups
    state_size = int(state_values.size)
    state_tasks = (state_size + 31) // 32
    workgroups = qk_tasks + v_tasks + state_tasks
    params = mx.array([float(eps)], dtype=mx.float32)
    outputs = _LINEAR_CONV_QKV_KERNEL(
        inputs=[
            state_values,
            qk_values,
            v_values,
            packed_weight,
            bias_values,
            params,
        ],
        template=[
            ("T", qk_values.dtype),
            ("B", batch),
            ("TOKENS", tokens),
            ("NK", nk),
            ("NV", nv),
            ("DK", dk),
            ("DV", dv),
            ("QKC", expected_qk),
            ("C", channels),
            ("K", kernel),
            ("HAS_BIAS", int(has_bias)),
        ],
        grid=(workgroups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, nk, tokens, dk),
            (batch, nk, tokens, dk),
            (batch, nv, tokens, dv),
            state_values.shape,
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return outputs[0], outputs[1], outputs[2], outputs[3]


__all__ = [
    "gated_delta_net",
    "linear_conv_qkv",
    "ssm_conv_silu",
]
