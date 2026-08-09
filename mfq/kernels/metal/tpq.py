"""Metal execution for TPQ dense-int4 and product-VQ weights.

The ``Tpq*`` symbols remain source-compatible aliases for artifacts created
before the public format was renamed to TPQ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.tpq import TpqInt4Tensor, TpqPqTensor
from mfq.formats.moe import NintMoeTensor

_PQ_INDEX_HEADER = r"""
template <typename I>
METAL_FUNC uint mfq_tpq_read_index(
    device const I* indices,
    uint index,
    uint bits
) {
    if (bits == 8u || bits == 16u) {
        return uint(indices[index]);
    }
    uint residual_bits = (index & 7u) * bits;
    uint byte_offset =
        (index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(indices[byte_offset]);
    if (shift + bits > 8u) {
        packed |= uint(indices[byte_offset + 1u]) << 8u;
    }
    if (shift + bits > 16u) {
        packed |= uint(indices[byte_offset + 2u]) << 16u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

METAL_FUNC uint mfq_tpq_read_packed_index(
    device const uchar* indices,
    uint byte_base,
    uint index,
    uint bits
) {
    uint residual_bits = (index & 7u) * bits;
    uint byte_offset = byte_base
        + (index >> 3) * bits
        + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(indices[byte_offset]);
    if (shift + bits > 8u) {
        packed |= uint(indices[byte_offset + 1u]) << 8u;
    }
    if (shift + bits > 16u) {
        packed |= uint(indices[byte_offset + 2u]) << 16u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

METAL_FUNC uint mfq_tpq_read_packed_index(
    constant const uchar* indices,
    uint byte_base,
    uint index,
    uint bits
) {
    uint residual_bits = (index & 7u) * bits;
    uint byte_offset = byte_base
        + (index >> 3) * bits
        + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(indices[byte_offset]);
    if (shift + bits > 8u) {
        packed |= uint(indices[byte_offset + 1u]) << 8u;
    }
    if (shift + bits > 16u) {
        packed |= uint(indices[byte_offset + 2u]) << 16u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}
"""

_INT4_MATMUL_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint row = task / uint(OUT);
    uint output = task % uint(OUT);
    if (row >= uint(ROWS)) {
        return;
    }
    uint weight_base = output * uint(K / 2);
    uint scale_base = output * uint(GROUPS);
    uint input_base = row * uint(K);
    float accumulator = 0.0f;
    for (uint column = lane * 2u; column < uint(K); column += 64u) {
        uint packed_value = uint(packed[weight_base + (column >> 1)]);
        float scale = float(scales[scale_base + column / uint(GROUP_SIZE)]);
        float low = float(int(packed_value & 15u) - 8) * scale;
        float high = float(int(packed_value >> 4u) - 8) * scale;
        accumulator = fma(low, float(x[input_base + column]), accumulator);
        accumulator = fma(
            high,
            float(x[input_base + column + 1u]),
            accumulator
        );
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[row * uint(OUT) + output] = T(accumulator);
    }
"""


_INT4_GROUPED_ROW_MATMUL_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint local_output = task % uint(OUT_PER_GROUP);
    uint group_row = task / uint(OUT_PER_GROUP);
    uint group = group_row % uint(GROUP_COUNT);
    uint row = group_row / uint(GROUP_COUNT);
    if (row >= uint(ROWS)) {
        return;
    }
    uint output = group * uint(OUT_PER_GROUP) + local_output;
    uint weight_base = output * uint(K / 2);
    uint scale_base = output * uint(SCALE_GROUPS);
    uint input_base = (
        row * uint(GROUP_COUNT) + group
    ) * uint(K);
    float accumulator = 0.0f;
    for (uint column = lane * 2u; column < uint(K); column += 64u) {
        uint packed_value = uint(packed[weight_base + (column >> 1)]);
        float scale = float(scales[
            scale_base + column / uint(GROUP_SIZE)
        ]);
        float low = float(int(packed_value & 15u) - 8) * scale;
        float high = float(int(packed_value >> 4u) - 8) * scale;
        accumulator = fma(
            low,
            float(x[input_base + column]),
            accumulator
        );
        accumulator = fma(
            high,
            float(x[input_base + column + 1u]),
            accumulator
        );
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[
            (row * uint(GROUP_COUNT) + group) * uint(OUT_PER_GROUP)
                + local_output
        ] = T(accumulator);
    }
"""


_INT4_DEQUANT_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(OUT * K)) {
        return;
    }
    uint row = index / uint(K);
    uint column = index - row * uint(K);
    uint packed_value = uint(packed[row * uint(K / 2) + (column >> 1)]);
    uint quantized = (column & 1u) == 0u
        ? packed_value & 15u
        : packed_value >> 4u;
    float scale = float(scales[
        row * uint(GROUPS) + column / uint(GROUP_SIZE)
    ]);
    output[index] = T(float(int(quantized) - 8) * scale);
"""


_INT4_EMBEDDING_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(ITEMS * K)) {
        return;
    }
    uint item = index / uint(K);
    uint column = index - item * uint(K);
    int row = int(ids[item]);
    if (row < 0 || row >= int(OUT)) {
        output[index] = T(0.0f);
        return;
    }
    uint packed_value = uint(packed[
        uint(row) * uint(K / 2) + (column >> 1)
    ]);
    uint quantized = (column & 1u) == 0u
        ? packed_value & 15u
        : packed_value >> 4u;
    float scale = float(scales[
        uint(row) * uint(GROUPS) + column / uint(GROUP_SIZE)
    ]);
    output[index] = T(float(int(quantized) - 8) * scale);
"""


_PQ_MATMUL_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint row = task / uint(OUT);
    uint output = task % uint(OUT);
    if (row >= uint(ROWS)) {
        return;
    }
    uint index_base = output * uint(BLOCKS);
    uint input_base = row * uint(K);
    float accumulator = 0.0f;
    for (uint block = lane; block < uint(BLOCKS); block += 32u) {
        uint code = mfq_tpq_read_index(
            indices,
            index_base + block,
            uint(INDEX_BITS)
        );
        uint code_base = code * uint(VECTOR_SIZE);
        uint column_base = block * uint(VECTOR_SIZE);
        for (uint component = 0u; component < uint(VECTOR_SIZE); ++component) {
            accumulator = fma(
                float(codebook[code_base + component]),
                float(x[input_base + column_base + component]),
                accumulator
            );
        }
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[row * uint(OUT) + output] = T(accumulator);
    }
"""


_PQ_ROUTED_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint output = task % uint(OUT);
    uint pair = task / uint(OUT);
    uint route = pair % uint(ROUTES);
    uint token = pair / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }
    int selected = int(selected_ids[token * uint(ROUTES) + route]);
    int local_expert = -1;
    for (uint candidate = 0u; candidate < uint(POOL_EXPERTS); ++candidate) {
        if (int(pool_ids[candidate]) == selected) {
            local_expert = int(candidate);
            break;
        }
    }
    uint destination = (
        (token * uint(ROUTES) + route) * uint(OUT) + output
    );
    if (local_expert < 0) {
        if (lane == 0u) {
            y[destination] = T(0.0f);
        }
        return;
    }
    uint packed_row = uint(local_expert) * uint(OUT) + output;
    uint index_base = packed_row * uint(BLOCKS);
    uint input_base = (
        token * uint(ROUTES) + route
    ) * uint(K);
    float accumulator = 0.0f;
    for (uint block = lane; block < uint(BLOCKS); block += 32u) {
        uint code = mfq_tpq_read_index(
            indices,
            index_base + block,
            uint(INDEX_BITS)
        );
        uint code_base = code * uint(VECTOR_SIZE);
        uint column_base = block * uint(VECTOR_SIZE);
        for (uint component = 0u; component < uint(VECTOR_SIZE); ++component) {
            accumulator = fma(
                float(codebook[code_base + component]),
                float(x[input_base + column_base + component]),
                accumulator
            );
        }
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[destination] = T(accumulator);
    }
"""


_PQ_DEQUANT_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(OUT * K)) {
        return;
    }
    uint row = index / uint(K);
    uint column = index - row * uint(K);
    uint block = column / uint(VECTOR_SIZE);
    uint component = column - block * uint(VECTOR_SIZE);
    uint code = mfq_tpq_read_index(
        indices,
        row * uint(BLOCKS) + block,
        uint(INDEX_BITS)
    );
    output[index] = T(codebook[code * uint(VECTOR_SIZE) + component]);
"""


_PQ_GROUPED_MOE_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint output = task % uint(OUT);
    uint pair = task / uint(OUT);
    uint route = pair % uint(ROUTES);
    uint token = pair / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }
    int expert = int(selected_ids[token * uint(ROUTES) + route]);
    uint destination = pair * uint(OUT) + output;
    if (expert < 0 || expert >= int(EXPERTS)) {
        if (lane == 0u) {
            y[destination] = T(0.0f);
        }
        return;
    }
    uint descriptor_base = uint(expert) * 6u;
    uint bits = uint(descriptors[descriptor_base]);
    uint local_expert = uint(descriptors[descriptor_base + 1u]);
    uint index_offset = uint(descriptors[descriptor_base + 2u]);
    uint codebook_offset = uint(descriptors[descriptor_base + 3u]);
    uint vector_size = uint(descriptors[descriptor_base + 4u]);
    uint blocks = uint(descriptors[descriptor_base + 5u]);
    uint packed_row = local_expert * uint(OUT) + output;
    uint row_offset = index_offset + packed_row * blocks;
    uint input_base = (
        SHARED_INPUT != 0
            ? token
            : token * uint(ROUTES) + route
    ) * uint(K);
    float accumulator = 0.0f;
    for (uint block = lane; block < blocks; block += 32u) {
        uint linear_index = packed_row * blocks + block;
        uint code;
        if (bits == 8u) {
            code = uint(indices8[index_offset + linear_index]);
        } else if (bits == 16u) {
            code = uint(indices16[index_offset + linear_index]);
        } else {
            code = mfq_tpq_read_packed_index(
                indices_packed,
                index_offset,
                linear_index,
                bits
            );
        }
        uint code_base = codebook_offset + code * vector_size;
        uint column_base = block * vector_size;
        for (uint component = 0u; component < vector_size; ++component) {
            accumulator = fma(
                float(codebooks[code_base + component]),
                float(x[input_base + column_base + component]),
                accumulator
            );
        }
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[destination] = T(accumulator);
    }
"""


_INT4_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_int4_matmul",
    input_names=["x", "packed", "scales"],
    output_names=["y"],
    source=_INT4_MATMUL_SOURCE,
    compile_options={"math_mode": "fast"},
)
_INT4_GROUPED_ROW_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_int4_grouped_row_matmul",
    input_names=["x", "packed", "scales"],
    output_names=["y"],
    source=_INT4_GROUPED_ROW_MATMUL_SOURCE,
    compile_options={"math_mode": "fast"},
)
_INT4_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_int4_dequantize",
    input_names=["packed", "scales"],
    output_names=["output"],
    source=_INT4_DEQUANT_SOURCE,
    compile_options={"math_mode": "fast"},
)
_INT4_EMBEDDING_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_int4_embedding",
    input_names=["packed", "scales", "ids"],
    output_names=["output"],
    source=_INT4_EMBEDDING_SOURCE,
    compile_options={"math_mode": "fast"},
)
_PQ_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_pq_matmul",
    input_names=["x", "indices", "codebook"],
    output_names=["y"],
    header=_PQ_INDEX_HEADER,
    source=_PQ_MATMUL_SOURCE,
    compile_options={"math_mode": "fast"},
)
_PQ_ROUTED_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_pq_routed_matmul",
    input_names=["x", "selected_ids", "pool_ids", "indices", "codebook"],
    output_names=["y"],
    header=_PQ_INDEX_HEADER,
    source=_PQ_ROUTED_SOURCE,
    compile_options={"math_mode": "fast"},
)
_PQ_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_pq_dequantize",
    input_names=["indices", "codebook"],
    output_names=["output"],
    header=_PQ_INDEX_HEADER,
    source=_PQ_DEQUANT_SOURCE,
    compile_options={"math_mode": "fast"},
)
_PQ_GROUPED_MOE_KERNEL = mx.fast.metal_kernel(
    name="mfq_tpq_pq_grouped_moe",
    input_names=[
        "descriptors",
        "indices8",
        "indices16",
        "indices_packed",
        "codebooks",
        "x",
        "selected_ids",
    ],
    output_names=["y"],
    header=_PQ_INDEX_HEADER,
    source=_PQ_GROUPED_MOE_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


@dataclass(frozen=True)
class MetalTpqInt4Weight:
    """Resident TPQ symmetric int4-g64 matrix."""

    packed: mx.array
    scales: mx.array
    out: int
    neuron_len: int
    group_size: int

    @classmethod
    def from_tensor(cls, tensor: TpqInt4Tensor) -> MetalTpqInt4Weight:
        return cls(
            packed=mx.array(np.ascontiguousarray(tensor.packed)),
            scales=mx.array(np.ascontiguousarray(tensor.scales)),
            out=int(tensor.shape[0]),
            neuron_len=int(tensor.shape[1]),
            group_size=int(tensor.group_size),
        )

    @property
    def groups(self) -> int:
        return self.neuron_len // self.group_size

    @property
    def packed_nbytes(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes)


@dataclass(frozen=True)
class MetalTpqPqWeight:
    """Resident TPQ2 product-VQ matrix with a shared FP32 codebook."""

    indices: mx.array
    codebook: mx.array
    out: int
    neuron_len: int
    vector_size: int
    entries: int
    index_bits: int

    @classmethod
    def from_tensor(cls, tensor: TpqPqTensor) -> MetalTpqPqWeight:
        bits = int(tensor.spec.index_bits)
        index_dtype = mx.uint16 if bits == 16 else mx.uint8
        return cls(
            indices=mx.array(np.ascontiguousarray(tensor.indices)).astype(index_dtype),
            codebook=mx.array(np.ascontiguousarray(tensor.codebook, dtype=np.float16)),
            out=int(tensor.shape[0]),
            neuron_len=int(tensor.shape[1]),
            vector_size=int(tensor.spec.vector_size),
            entries=int(tensor.spec.codebook_entries),
            index_bits=bits,
        )

    @property
    def blocks(self) -> int:
        return self.neuron_len // self.vector_size

    @property
    def packed_nbytes(self) -> int:
        return int(self.indices.nbytes + self.codebook.nbytes)


@dataclass(frozen=True)
class MetalTpqMoeWeight:
    """Single-dispatch heterogeneous TPQ expert projection."""

    descriptors: mx.array
    indices8: mx.array
    indices16: mx.array
    indices_packed: mx.array
    codebooks: mx.array
    experts: int
    out_per_expert: int
    neuron_len: int

    @classmethod
    def from_tensor(cls, tensor: NintMoeTensor) -> MetalTpqMoeWeight:
        descriptors = np.zeros((tensor.n_experts, 6), dtype=np.int32)
        streams8: list[mx.array] = []
        streams16: list[mx.array] = []
        streams_packed: list[mx.array] = []
        tables: list[mx.array] = []
        offset8 = 0
        offset16 = 0
        offset_packed = 0
        table_offset = 0
        for pool in tensor.pools:
            if not isinstance(pool.tensor, TpqPqTensor):
                raise TypeError("MetalTpqMoeWeight requires exclusively TPQ-PQ cohorts")
            weight = MetalTpqPqWeight.from_tensor(pool.tensor)
            ids = np.asarray(pool.expert_ids, dtype=np.int32).reshape((-1,))
            expected_rows = int(ids.size) * tensor.out_per_expert
            if weight.out != expected_rows:
                raise ValueError("TPQ MoE cohort row count is inconsistent")
            bits = int(pool.tensor.spec.index_bits)
            if bits == 8:
                index_offset = offset8
            elif bits == 16:
                index_offset = offset16
            else:
                index_offset = offset_packed
            for local_expert, expert in enumerate(ids.tolist()):
                descriptors[expert] = (
                    bits,
                    local_expert,
                    index_offset,
                    table_offset,
                    weight.vector_size,
                    weight.blocks,
                )
            if bits == 8:
                streams8.append(weight.indices.reshape((-1,)))
                offset8 += int(weight.indices.size)
            elif bits == 16:
                streams16.append(weight.indices.reshape((-1,)))
                offset16 += int(weight.indices.size)
            else:
                streams_packed.append(weight.indices.reshape((-1,)))
                offset_packed += int(weight.indices.size)
            tables.append(weight.codebook.reshape((-1,)))
            table_offset += int(weight.codebook.size)

        def join(values: list[mx.array], dtype: mx.Dtype) -> mx.array:
            if not values:
                return mx.zeros((1,), dtype=dtype)
            return mx.contiguous(mx.concatenate(values).astype(dtype))

        return cls(
            descriptors=mx.array(descriptors),
            indices8=join(streams8, mx.uint8),
            indices16=join(streams16, mx.uint16),
            indices_packed=join(streams_packed, mx.uint8),
            codebooks=join(tables, mx.float16),
            experts=tensor.n_experts,
            out_per_expert=tensor.out_per_expert,
            neuron_len=tensor.neuron_len,
        )

    @classmethod
    def from_expert_weights(
        cls,
        weights: tuple[tuple[int, MetalTpqPqWeight], ...],
        *,
        experts: int,
        out_per_expert: int,
        neuron_len: int,
    ) -> MetalTpqMoeWeight:
        """Assemble one transient grouped weight from resident expert slices."""

        descriptors = np.zeros((int(experts), 6), dtype=np.int32)
        streams8: list[mx.array] = []
        streams16: list[mx.array] = []
        streams_packed: list[mx.array] = []
        tables: list[mx.array] = []
        offsets = {8: 0, 16: 0, -1: 0}
        table_offset = 0
        seen: set[int] = set()
        for expert, weight in weights:
            expert_id = int(expert)
            if (
                expert_id in seen
                or not 0 <= expert_id < int(experts)
                or weight.out != int(out_per_expert)
                or weight.neuron_len != int(neuron_len)
            ):
                raise ValueError("resident TPQ expert weight metadata is inconsistent")
            seen.add(expert_id)
            bits = int(weight.index_bits)
            stream_key = bits if bits in (8, 16) else -1
            index_offset = offsets[stream_key]
            descriptors[expert_id] = (
                bits,
                0,
                index_offset,
                table_offset,
                weight.vector_size,
                weight.blocks,
            )
            if bits == 8:
                streams8.append(weight.indices.reshape((-1,)))
            elif bits == 16:
                streams16.append(weight.indices.reshape((-1,)))
            else:
                streams_packed.append(weight.indices.reshape((-1,)))
            offsets[stream_key] += int(weight.indices.size)
            tables.append(weight.codebook.reshape((-1,)))
            table_offset += int(weight.codebook.size)

        def join(values: list[mx.array], dtype: mx.Dtype) -> mx.array:
            if not values:
                return mx.zeros((1,), dtype=dtype)
            return mx.contiguous(mx.concatenate(values).astype(dtype))

        return cls(
            descriptors=mx.array(descriptors),
            indices8=join(streams8, mx.uint8),
            indices16=join(streams16, mx.uint16),
            indices_packed=join(streams_packed, mx.uint8),
            codebooks=join(tables, mx.float16),
            experts=int(experts),
            out_per_expert=int(out_per_expert),
            neuron_len=int(neuron_len),
        )

    @property
    def packed_nbytes(self) -> int:
        return int(
            self.descriptors.nbytes
            + self.indices8.nbytes
            + self.indices16.nbytes
            + self.indices_packed.nbytes
            + self.codebooks.nbytes
        )


def tpq_int4_matmul(
    weight: MetalTpqInt4Weight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Apply a packed TPQ2 dense int4 projection without dequantizing it."""

    source = _floating(x)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError("TPQ int4 input width is incompatible")
    rows = int(source.size) // weight.neuron_len
    outputs = _INT4_MATMUL_KERNEL(
        inputs=[
            source.reshape((rows, weight.neuron_len)),
            weight.packed,
            weight.scales,
        ],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("GROUP_SIZE", weight.group_size),
            ("GROUPS", weight.groups),
        ],
        grid=(rows * weight.out * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*map(int, source.shape[:-1]), weight.out)],
        output_dtypes=[source.dtype],
    )
    return outputs[0]


def tpq_int4_grouped_row_matmul(
    weight: MetalTpqInt4Weight,
    x: mx.array | np.ndarray,
    *,
    groups: int,
) -> mx.array:
    """Apply disjoint weight-row groups to matching input groups.

    ``x[..., g, :]`` is multiplied only by weight rows belonging to group
    ``g``.  DeepSeek-V4 uses this layout for the eight independent ``wo_a``
    low-rank projections.
    """

    source = _floating(x)
    group_count = int(groups)
    if (
        source.ndim < 2
        or group_count <= 0
        or int(source.shape[-2]) != group_count
        or int(source.shape[-1]) != weight.neuron_len
        or weight.out % group_count
    ):
        raise ValueError("TPQ grouped-row int4 input is incompatible")
    rows = int(source.size) // (group_count * weight.neuron_len)
    out_per_group = weight.out // group_count
    return _INT4_GROUPED_ROW_MATMUL_KERNEL(
        inputs=[
            source.reshape((rows, group_count, weight.neuron_len)),
            weight.packed,
            weight.scales,
        ],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("GROUP_COUNT", group_count),
            ("OUT_PER_GROUP", out_per_group),
            ("K", weight.neuron_len),
            ("GROUP_SIZE", weight.group_size),
            ("SCALE_GROUPS", weight.groups),
        ],
        grid=(rows * group_count * out_per_group * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*map(int, source.shape[:-2]), group_count, out_per_group)],
        output_dtypes=[source.dtype],
    )[0]


def tpq_int4_dequantize(
    weight: MetalTpqInt4Weight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Materialize a TPQ int4 matrix for validation or weight absorption."""

    size = weight.out * weight.neuron_len
    return _INT4_DEQUANT_KERNEL(
        inputs=[weight.packed, weight.scales],
        template=[
            ("T", dtype),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("GROUP_SIZE", weight.group_size),
            ("GROUPS", weight.groups),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(weight.out, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]


def tpq_int4_embedding(
    weight: MetalTpqInt4Weight,
    ids: mx.array | np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode only the requested TPQ int4 embedding rows."""

    selected = ids if isinstance(ids, mx.array) else mx.array(ids)
    selected = mx.contiguous(selected.astype(mx.int32))
    items = int(selected.size)
    size = items * weight.neuron_len
    return _INT4_EMBEDDING_KERNEL(
        inputs=[weight.packed, weight.scales, selected],
        template=[
            ("T", dtype),
            ("ITEMS", items),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("GROUP_SIZE", weight.group_size),
            ("GROUPS", weight.groups),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, max(1, size)), 1, 1),
        output_shapes=[(*map(int, selected.shape), weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]


def tpq_pq_matmul(
    weight: MetalTpqPqWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Apply a TPQ2 codebook matrix without materializing dense weights."""

    source = _floating(x)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError("TPQ-PQ input width is incompatible")
    rows = int(source.size) // weight.neuron_len
    return _PQ_MATMUL_KERNEL(
        inputs=[
            source.reshape((rows, weight.neuron_len)),
            weight.indices,
            weight.codebook,
        ],
        template=[
            ("T", source.dtype),
            ("IDX", weight.indices.dtype),
            ("INDEX_BITS", weight.index_bits),
            ("ROWS", rows),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("BLOCKS", weight.blocks),
            ("VECTOR_SIZE", weight.vector_size),
        ],
        grid=(rows * weight.out * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*map(int, source.shape[:-1]), weight.out)],
        output_dtypes=[source.dtype],
    )[0]


def tpq_pq_routed_matmul(
    weight: MetalTpqPqWeight,
    x: mx.array | np.ndarray,
    selected_ids: mx.array | np.ndarray,
    pool_ids: mx.array | np.ndarray,
    *,
    out_per_expert: int,
) -> mx.array:
    """Evaluate only selected expert rows from one TPQ-PQ cohort."""

    source = _floating(x)
    ids = selected_ids if isinstance(selected_ids, mx.array) else mx.array(selected_ids)
    ids = mx.contiguous(ids.astype(mx.int32))
    experts = pool_ids if isinstance(pool_ids, mx.array) else mx.array(pool_ids)
    experts = mx.contiguous(experts.astype(mx.int32).reshape((-1,)))
    if ids.ndim != 2:
        raise ValueError("TPQ routed IDs must have [tokens,routes] shape")
    tokens, routes = map(int, ids.shape)
    output_width = int(out_per_expert)
    if source.ndim == 2 and tuple(map(int, source.shape)) == (tokens, weight.neuron_len):
        source = mx.broadcast_to(
            source[:, None, :],
            (tokens, routes, weight.neuron_len),
        )
    if tuple(map(int, source.shape)) != (tokens, routes, weight.neuron_len):
        raise ValueError("TPQ routed input has incompatible shape")
    if weight.out != int(experts.size) * output_width:
        raise ValueError("TPQ routed cohort rows do not match pool expert IDs")
    return _PQ_ROUTED_KERNEL(
        inputs=[source, ids, experts, weight.indices, weight.codebook],
        template=[
            ("T", source.dtype),
            ("IDX", weight.indices.dtype),
            ("INDEX_BITS", weight.index_bits),
            ("TOKENS", tokens),
            ("ROUTES", routes),
            ("POOL_EXPERTS", int(experts.size)),
            ("OUT", output_width),
            ("K", weight.neuron_len),
            ("BLOCKS", weight.blocks),
            ("VECTOR_SIZE", weight.vector_size),
        ],
        grid=(tokens * routes * output_width * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(tokens, routes, output_width)],
        output_dtypes=[source.dtype],
    )[0]


def tpq_pq_dequantize(
    weight: MetalTpqPqWeight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Materialize a TPQ2 codebook matrix for validation."""

    size = weight.out * weight.neuron_len
    return _PQ_DEQUANT_KERNEL(
        inputs=[weight.indices, weight.codebook],
        template=[
            ("T", dtype),
            ("IDX", weight.indices.dtype),
            ("INDEX_BITS", weight.index_bits),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("BLOCKS", weight.blocks),
            ("VECTOR_SIZE", weight.vector_size),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(weight.out, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]


def tpq_grouped_moe_matmul(
    weight: MetalTpqMoeWeight,
    x: mx.array | np.ndarray,
    selected_ids: mx.array | np.ndarray,
) -> mx.array:
    """Evaluate heterogeneous TPQ2 expert cohorts in one Metal dispatch."""

    source = _floating(x)
    ids = selected_ids if isinstance(selected_ids, mx.array) else mx.array(selected_ids)
    ids = mx.contiguous(ids.astype(mx.int32))
    if ids.ndim != 2:
        raise ValueError("TPQ grouped IDs must have [tokens,routes] shape")
    tokens, routes = map(int, ids.shape)
    shared_input = source.ndim == 2 and tuple(map(int, source.shape)) == (tokens, weight.neuron_len)
    if not shared_input and tuple(map(int, source.shape)) != (
        tokens,
        routes,
        weight.neuron_len,
    ):
        raise ValueError("TPQ grouped input has incompatible shape")
    tasks = tokens * routes * weight.out_per_expert
    return _PQ_GROUPED_MOE_KERNEL(
        inputs=[
            weight.descriptors,
            weight.indices8,
            weight.indices16,
            weight.indices_packed,
            weight.codebooks,
            source,
            ids,
        ],
        template=[
            ("T", source.dtype),
            ("TOKENS", tokens),
            ("ROUTES", routes),
            ("EXPERTS", weight.experts),
            ("OUT", weight.out_per_expert),
            ("K", weight.neuron_len),
            ("SHARED_INPUT", int(shared_input)),
        ],
        grid=(tasks * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(tokens, routes, weight.out_per_expert)],
        output_dtypes=[source.dtype],
    )[0]


__all__ = [
    "MetalTpqInt4Weight",
    "MetalTpqMoeWeight",
    "MetalTpqPqWeight",
    "tpq_grouped_moe_matmul",
    "tpq_int4_dequantize",
    "tpq_int4_embedding",
    "tpq_int4_grouped_row_matmul",
    "tpq_int4_matmul",
    "tpq_pq_dequantize",
    "tpq_pq_matmul",
    "tpq_pq_routed_matmul",
]
