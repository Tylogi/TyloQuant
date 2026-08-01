"""GPU-resident logits sampling and penalty kernels for Apple silicon."""

from __future__ import annotations

import math

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_GREEDY_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float values[256];
    threadgroup int indices[256];
    float best = -FLT_MAX;
    int best_index = 0;
    uint offset = row * uint(VOCAB);
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]);
        value = isnan(value) ? -FLT_MAX : value;
        if (value > best || (value == best && int(token) < best_index)) {
            best = value;
            best_index = int(token);
        }
    }
    values[tid] = best;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other = values[tid + stride];
            int other_index = indices[tid + stride];
            if (
                other > values[tid]
                || (other == values[tid] && other_index < indices[tid])
            ) {
                values[tid] = other;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        output[row] = indices[0];
    }
"""


_SOFTMAX_SAMPLE_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float reduction[256];
    threadgroup float maximum;
    threadgroup float denominator;
    uint offset = row * uint(VOCAB);
    float local_max = -FLT_MAX;
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        local_max = max(local_max, value);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] = max(reduction[tid], reduction[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        maximum = reduction[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        local_sum += exp(value - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        denominator = reduction[0];
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * denominator;
        float cumulative = 0.0f;
        int chosen = int(VOCAB) - 1;
        for (uint token = 0u; token < uint(VOCAB); ++token) {
            float value = float(logits[offset + token]) / params[0];
            value = isnan(value) ? -FLT_MAX : value;
            cumulative += exp(value - maximum);
            if (cumulative >= target) {
                chosen = int(token);
                break;
            }
        }
        output[row] = chosen;
    }
"""


_TOP_K_SAMPLE_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float top_values[TOP_K];
    threadgroup int top_indices[TOP_K];
    threadgroup float reduction_values[256];
    threadgroup int reduction_indices[256];
    threadgroup float probabilities[TOP_K];
    uint offset = row * uint(VOCAB);

    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        float best = -FLT_MAX;
        int best_index = int(VOCAB);
        for (uint token = tid; token < uint(VOCAB); token += 256u) {
            bool selected = false;
            for (uint previous = 0u; previous < rank; ++previous) {
                selected = selected || top_indices[previous] == int(token);
            }
            if (selected) {
                continue;
            }
            float value = float(logits[offset + token]) / params[0];
            value = isnan(value) ? -FLT_MAX : value;
            if (
                value > best
                || (value == best && int(token) < best_index)
            ) {
                best = value;
                best_index = int(token);
            }
        }
        reduction_values[tid] = best;
        reduction_indices[tid] = best_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float other = reduction_values[tid + stride];
                int other_index = reduction_indices[tid + stride];
                if (
                    other > reduction_values[tid]
                    || (
                        other == reduction_values[tid]
                        && other_index < reduction_indices[tid]
                    )
                ) {
                    reduction_values[tid] = other;
                    reduction_indices[tid] = other_index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            top_values[rank] = reduction_values[0];
            top_indices[rank] = reduction_indices[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float maximum = top_values[0];
        float total = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float probability = exp(top_values[rank] - maximum);
            probabilities[rank] = probability;
            total += probability;
        }
        uint keep = uint(TOP_K);
        float keep_sum = total;
        if (params[1] > 0.0f && params[1] < 1.0f) {
            float cutoff = params[1] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                cumulative += probabilities[rank];
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = top_indices[keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += probabilities[rank];
            if (cumulative >= target) {
                chosen = top_indices[rank];
                break;
            }
        }
        output[row] = chosen;
    }
"""


_SORTED_SAMPLE_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float reduction[256];
    threadgroup float total;
    uint offset = row * uint(COUNT);
    float maximum = scores[offset];
    float local_sum = 0.0f;
    for (uint rank = tid; rank < uint(COUNT); rank += 256u) {
        local_sum += exp(scores[offset + rank] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        total = reduction[0];
        uint keep = uint(COUNT);
        float keep_sum = total;
        if (params[0] > 0.0f && params[0] < 1.0f) {
            float cutoff = params[0] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(COUNT); ++rank) {
                cumulative += exp(scores[offset + rank] - maximum);
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = indices[offset + keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += exp(scores[offset + rank] - maximum);
            if (cumulative >= target) {
                chosen = indices[offset + rank];
                break;
            }
        }
        output[row] = chosen;
    }
"""


_TOKEN_COUNTS_SOURCE = r"""
    uint token = thread_position_in_grid.x;
    if (token >= uint(VOCAB)) {
        return;
    }
    int value = counts[token];
    for (uint index = 0u; index < uint(TOKENS); ++index) {
        value += int(token_ids[index] == int(token));
    }
    output[token] = value;
"""


_PENALTIES_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint token = index % uint(VOCAB);
    int count = counts[token];
    float value = float(logits[index]);
    if (count > 0) {
        if (params[2] != 1.0f) {
            value = value < 0.0f
                ? value * params[2]
                : value / params[2];
        }
        value -= params[0] + params[1] * float(count);
    }
    output[index] = T(value);
"""


_GREEDY_KERNEL = mx.fast.metal_kernel(
    name="mfq_sample_greedy",
    input_names=["logits"],
    output_names=["output"],
    source=_GREEDY_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SOFTMAX_SAMPLE_KERNEL = mx.fast.metal_kernel(
    name="mfq_sample_softmax",
    input_names=["logits", "random", "params"],
    output_names=["output"],
    source=_SOFTMAX_SAMPLE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_TOP_K_SAMPLE_KERNEL = mx.fast.metal_kernel(
    name="mfq_sample_top_k_top_p",
    input_names=["logits", "random", "params"],
    output_names=["output"],
    source=_TOP_K_SAMPLE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SORTED_SAMPLE_KERNEL = mx.fast.metal_kernel(
    name="mfq_sample_sorted_top_p",
    input_names=["scores", "indices", "random", "params"],
    output_names=["output"],
    source=_SORTED_SAMPLE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_TOKEN_COUNTS_KERNEL = mx.fast.metal_kernel(
    name="mfq_sample_token_counts_add",
    input_names=["counts", "token_ids"],
    output_names=["output"],
    source=_TOKEN_COUNTS_SOURCE,
)

_PENALTIES_KERNEL = mx.fast.metal_kernel(
    name="mfq_sample_apply_penalties",
    input_names=["logits", "counts", "params"],
    output_names=["output"],
    source=_PENALTIES_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _logits(value: mx.array | np.ndarray) -> tuple[mx.array, tuple[int, ...], int, int]:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.ndim < 1 or int(result.shape[-1]) <= 0:
        raise ValueError("sampling logits must end in a non-empty vocabulary")
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float32)
    result = mx.contiguous(result)
    shape = tuple(int(item) for item in result.shape)
    vocab = shape[-1]
    rows = int(result.size) // vocab
    return result.reshape((rows, vocab)), shape[:-1], rows, vocab


def _random_values(
    value: mx.array | np.ndarray | None,
    rows: int,
) -> mx.array:
    if value is None:
        result = mx.random.uniform(shape=(rows,))
    else:
        result = value if isinstance(value, mx.array) else mx.array(value)
    result = mx.contiguous(result.astype(mx.float32).reshape((-1,)))
    if int(result.size) != rows:
        raise ValueError("sampling random values must contain one item per row")
    return result


def sample_greedy(logits: mx.array | np.ndarray) -> mx.array:
    """Argmax the final dimension entirely on Metal."""

    values, prefix, rows, vocab = _logits(logits)
    result = _GREEDY_KERNEL(
        inputs=[values],
        template=[("T", values.dtype), ("ROWS", rows), ("VOCAB", vocab)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )[0]
    return result.reshape(prefix)


def sample_softmax(
    logits: mx.array | np.ndarray,
    random: mx.array | np.ndarray | None = None,
    *,
    temperature: float = 1.0,
) -> mx.array:
    """Sample a full-vocabulary softmax with a supplied or device RNG uniform."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values, prefix, rows, vocab = _logits(logits)
    uniforms = _random_values(random, rows)
    params = mx.array([float(temperature)], dtype=mx.float32)
    result = _SOFTMAX_SAMPLE_KERNEL(
        inputs=[values, uniforms, params],
        template=[("T", values.dtype), ("ROWS", rows), ("VOCAB", vocab)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )[0]
    return result.reshape(prefix)


def _sample_sorted(
    values: mx.array,
    uniforms: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mx.array:
    rows, vocab = (int(item) for item in values.shape)
    scores = values.astype(mx.float32) / float(temperature)
    scores = mx.where(mx.isnan(scores), -mx.inf, scores)
    order = mx.argsort(scores, axis=-1)[:, ::-1]
    count = vocab if top_k <= 0 else min(int(top_k), vocab)
    order = mx.contiguous(order[:, :count].astype(mx.int32))
    ordered = mx.contiguous(mx.take_along_axis(scores, order, axis=-1))
    params = mx.array([float(top_p)], dtype=mx.float32)
    return _SORTED_SAMPLE_KERNEL(
        inputs=[ordered, order, uniforms, params],
        template=[("ROWS", rows), ("COUNT", count)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )[0]


def sample_top_k_top_p(
    logits: mx.array | np.ndarray,
    random: mx.array | np.ndarray | None = None,
    *,
    temperature: float = 1.0,
    top_k: int,
    top_p: float = 1.0,
) -> mx.array:
    """Sample within top-k, applying nucleus truncation inside that set."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0,1]")
    values, prefix, rows, vocab = _logits(logits)
    selected = int(top_k)
    if not 1 <= selected <= min(vocab, 1024):
        raise ValueError("top_k must be in [1,min(vocab,1024)]")
    uniforms = _random_values(random, rows)
    if selected > 64:
        result = _sample_sorted(
            values,
            uniforms,
            temperature=float(temperature),
            top_k=selected,
            top_p=float(top_p),
        )
    else:
        params = mx.array(
            [float(temperature), float(top_p)],
            dtype=mx.float32,
        )
        result = _TOP_K_SAMPLE_KERNEL(
            inputs=[values, uniforms, params],
            template=[
                ("T", values.dtype),
                ("ROWS", rows),
                ("VOCAB", vocab),
                ("TOP_K", selected),
            ],
            grid=(rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows,)],
            output_dtypes=[mx.int32],
        )[0]
    return result.reshape(prefix)


def sample(
    logits: mx.array | np.ndarray,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    random: mx.array | np.ndarray | None = None,
) -> mx.array:
    """CUDA-compatible sampling with global nucleus support for ``top_k=0``."""

    if int(top_k) < 0:
        raise ValueError("top_k must be non-negative")
    if temperature <= 0.0 or int(top_k) == 1:
        return sample_greedy(logits)
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0,1]")
    if int(top_k) > 0:
        return sample_top_k_top_p(
            logits,
            random,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )
    if top_p >= 1.0:
        return sample_softmax(
            logits,
            random,
            temperature=float(temperature),
        )
    values, prefix, rows, _vocab = _logits(logits)
    uniforms = _random_values(random, rows)
    result = _sample_sorted(
        values,
        uniforms,
        temperature=float(temperature),
        top_k=0,
        top_p=float(top_p),
    )
    return result.reshape(prefix)


def sample_token_counts_add(
    counts: mx.array | np.ndarray,
    tokens: mx.array | np.ndarray,
) -> mx.array:
    """Add token occurrences to one int32 vocabulary-count vector."""

    current = counts if isinstance(counts, mx.array) else mx.array(counts)
    current = mx.contiguous(current.astype(mx.int32).reshape((-1,)))
    token_ids = tokens if isinstance(tokens, mx.array) else mx.array(tokens)
    token_ids = mx.contiguous(token_ids.astype(mx.int32).reshape((-1,)))
    vocab = int(current.size)
    if vocab <= 0:
        raise ValueError("token counts cannot be empty")
    return _TOKEN_COUNTS_KERNEL(
        inputs=[current, token_ids],
        template=[("VOCAB", vocab), ("TOKENS", int(token_ids.size))],
        grid=(vocab, 1, 1),
        threadgroup=(min(256, vocab), 1, 1),
        output_shapes=[(vocab,)],
        output_dtypes=[mx.int32],
    )[0]


def sample_apply_penalties(
    logits: mx.array | np.ndarray,
    counts: mx.array | np.ndarray,
    *,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> mx.array:
    """Apply OpenAI-style presence/frequency and sign-aware repetition penalties."""

    values, prefix, rows, vocab = _logits(logits)
    token_counts = counts if isinstance(counts, mx.array) else mx.array(counts)
    token_counts = mx.contiguous(token_counts.astype(mx.int32).reshape((-1,)))
    if int(token_counts.size) != vocab:
        raise ValueError("penalty counts must match the vocabulary size")
    parameters = (
        float(presence_penalty),
        float(frequency_penalty),
        float(repetition_penalty),
    )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("sampling penalties must be finite")
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    params = mx.array(parameters, dtype=mx.float32)
    size = rows * vocab
    output = _PENALTIES_KERNEL(
        inputs=[values, token_counts, params],
        template=[("T", values.dtype), ("SIZE", size), ("VOCAB", vocab)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(rows, vocab)],
        output_dtypes=[values.dtype],
    )[0]
    return output.reshape((*prefix, vocab))


__all__ = [
    "sample",
    "sample_apply_penalties",
    "sample_greedy",
    "sample_softmax",
    "sample_token_counts_add",
    "sample_top_k_top_p",
]
