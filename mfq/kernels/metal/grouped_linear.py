"""Single-dispatch heterogeneous linear projections for Apple silicon.

The decode path concatenates packed NINT, NINT8-0, VQ-family, and TPQ
streams once. One fixed-width descriptor selects the decoder for each
projection, allowing Q/K/V or gate/up matrices with different output widths
and formats to share one Metal dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.kernels.metal.tpq import (
    _PQ_INDEX_HEADER,
    MetalTpqInt4Weight,
    MetalTpqPqWeight,
)
from mfq.kernels.metal.moe import (
    _DESCRIPTOR_SIZE,
    _FAMILY,
    _FAMILY_NINT,
    _FAMILY_NINT8_ZERO,
    _FAMILY_VQ,
    _GROUPED_HEADER,
    _K,
    _LOCAL_EXPERT,
    _NINT_ANCHOR_OFFSET,
    _NINT_BITS,
    _NINT_GS,
    _NINT_NG,
    _NINT_Q5_EXEC,
    _NINT_Q_OFFSET,
    _NINT_SUB_OFFSET,
    _OUT,
    _Q8_NG,
    _Q8_Q_OFFSET,
    _Q8_SCALE_OFFSET,
    _VQ_ANCHOR_OFFSET,
    _VQ_AUX_MODE,
    _VQ_AUX_OFFSET,
    _VQ_BANK_OFFSET,
    _VQ_CODE_BANK_MODE,
    _VQ_CODE_BANKS,
    _VQ_CODEBOOK_OFFSET,
    _VQ_ENTRIES,
    _VQ_GROUPS_PER_SUPER,
    _VQ_GS,
    _VQ_HAS_TABLE_BANKS,
    _VQ_INDEX_BITS,
    _VQ_INDICES_OFFSET,
    _VQ_NG,
    _VQ_NSUPER,
    _VQ_NVEC,
    _VQ_PARAMETER_OFFSET,
    _VQ_ROTATION_VARIANT,
    _VQ_SCALE_OFFSET,
    _VQ_STATE_BANK_OFFSET,
    _VQ_STATE_BITS,
    _VQ_STATE_OFFSET,
    _VQ_STATES,
    _VQ_VECTOR_SIZE,
    _join,
    _size,
)
from mfq.kernels.metal.nint import MetalNintWeight
from mfq.kernels.metal.nint8_zero import MetalNint8ZeroWeight
from mfq.kernels.metal.vq import MetalVqWeight, signed_hadamard

_FAMILY_TPQ_INT4 = 3
_FAMILY_TPQ_PQ = 4

# TPQ int4 descriptor fields.
_TPQ_I4_GROUP_SIZE = 4
_TPQ_I4_GROUPS = 5
_TPQ_I4_PACKED_OFFSET = 6
_TPQ_I4_SCALE_OFFSET = 7

# TPQ product-VQ descriptor fields.
_TPQ_PQ_BITS = 4
_TPQ_PQ_VECTOR_SIZE = 5
_TPQ_PQ_BLOCKS = 6
_TPQ_PQ_INDEX_OFFSET = 7
_TPQ_PQ_CODEBOOK_OFFSET = 8

PackedLinearWeight: TypeAlias = (
    MetalNintWeight | MetalNint8ZeroWeight | MetalVqWeight | MetalTpqInt4Weight | MetalTpqPqWeight
)


_GROUPED_LINEAR_SOURCE = r"""
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile = workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    while (
        projection + 1u < uint(PROJECTIONS)
        && global_tile >= uint(projection_tile_offsets[projection + 1u])
    ) {
        ++projection;
    }
    uint local_tile =
        global_tile - uint(projection_tile_offsets[projection]);
    uint descriptor_base = projection * uint(DESCRIPTOR_SIZE);
    uint output_width = uint(descriptors[descriptor_base + 2u]);
    uint output_base =
        local_tile * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;
    uint output_offset = uint(projection_output_offsets[projection]);
    uint family = uint(descriptors[descriptor_base]);
    uint rotation_variant = uint(descriptors[descriptor_base + 27u]);
    uint input_base = (
        rotation_variant * uint(ROWS) + input_row
    ) * uint(K);

    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation = float(x[input_base + column]);
        for (uint local_row = 0u; local_row < ROWS_PER_SIMD; ++local_row) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }
            float weight;
            if (family == 3u) {
                uint group_size =
                    uint(descriptors[descriptor_base + 4u]);
                uint groups = uint(descriptors[descriptor_base + 5u]);
                uint packed_offset =
                    uint(descriptors[descriptor_base + 6u]);
                uint scale_offset =
                    uint(descriptors[descriptor_base + 7u]);
                uint packed_value = uint(tpq_i4_packed[
                    packed_offset + output * uint(K / 2u) + (column >> 1)
                ]);
                uint quantized = (column & 1u) == 0u
                    ? packed_value & 15u
                    : packed_value >> 4u;
                float scale = float(tpq_i4_scales[
                    scale_offset
                    + output * groups
                    + column / group_size
                ]);
                weight = float(int(quantized) - 8) * scale;
            } else if (family == 4u) {
                uint bits = uint(descriptors[descriptor_base + 4u]);
                uint vector_size =
                    uint(descriptors[descriptor_base + 5u]);
                uint blocks = uint(descriptors[descriptor_base + 6u]);
                uint index_offset =
                    uint(descriptors[descriptor_base + 7u]);
                uint codebook_offset =
                    uint(descriptors[descriptor_base + 8u]);
                uint block = column / vector_size;
                uint component = column - block * vector_size;
                uint linear_index = output * blocks + block;
                uint code;
                if (bits == 8u) {
                    code = uint(
                        tpq_pq_indices8[index_offset + linear_index]
                    );
                } else if (bits == 16u) {
                    code = uint(
                        tpq_pq_indices16[index_offset + linear_index]
                    );
                } else {
                    code = mfq_tpq_read_packed_index(
                        tpq_pq_indices_packed,
                        index_offset,
                        linear_index,
                        bits
                    );
                }
                weight = float(tpq_pq_codebooks[
                    codebook_offset + code * vector_size + component
                ]);
            } else {
                weight = mfq_grouped_decode_weight(
                    descriptors,
                    nint_q,
                    nint_sub_scale,
                    nint_sub_min,
                    nint_anchor_scale,
                    nint_anchor_min,
                    q8_q,
                    q8_scales,
                    vq_indices,
                    vq_state,
                    vq_aux,
                    vq_anchors,
                    vq_codebooks,
                    vq_scales,
                    vq_state_to_codebank,
                    vq_banks,
                    vq_parameters,
                    descriptor_base,
                    output,
                    column,
                    uint(K)
                );
            }
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]
            );
        }
    }

    for (uint local_row = 0u; local_row < ROWS_PER_SIMD; ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
"""


_GROUPED_LINEAR_KERNEL = mx.fast.metal_kernel(
    name="mfq_heterogeneous_grouped_linear",
    input_names=[
        "descriptors",
        "projection_tile_offsets",
        "projection_output_offsets",
        "nint_q",
        "nint_sub_scale",
        "nint_sub_min",
        "nint_anchor_scale",
        "nint_anchor_min",
        "q8_q",
        "q8_scales",
        "vq_indices",
        "vq_state",
        "vq_aux",
        "vq_anchors",
        "vq_codebooks",
        "vq_scales",
        "vq_state_to_codebank",
        "vq_banks",
        "vq_parameters",
        "tpq_i4_packed",
        "tpq_i4_scales",
        "tpq_pq_indices8",
        "tpq_pq_indices16",
        "tpq_pq_indices_packed",
        "tpq_pq_codebooks",
        "x",
    ],
    output_names=["y"],
    header=_GROUPED_HEADER + _PQ_INDEX_HEADER,
    source=_GROUPED_LINEAR_SOURCE,
    compile_options={"math_mode": "fast"},
)


@dataclass(frozen=True)
class MetalLinearGroupWeight:
    """Concatenated packed buffers for ordinary heterogeneous projections."""

    descriptors: mx.array
    projection_tile_offsets: mx.array
    projection_output_offsets: mx.array
    nint_q: mx.array
    nint_sub_scale: mx.array
    nint_sub_min: mx.array
    nint_anchor_scale: mx.array
    nint_anchor_min: mx.array
    q8_q: mx.array
    q8_scales: mx.array
    vq_indices: mx.array
    vq_state: mx.array
    vq_aux: mx.array
    vq_anchors: mx.array
    vq_codebooks: mx.array
    vq_scales: mx.array
    vq_state_to_codebank: mx.array
    vq_banks: mx.array
    vq_parameters: mx.array
    tpq_i4_packed: mx.array
    tpq_i4_scales: mx.array
    tpq_pq_indices8: mx.array
    tpq_pq_indices16: mx.array
    tpq_pq_indices_packed: mx.array
    tpq_pq_codebooks: mx.array
    descriptor_values: np.ndarray
    rotation_specs: tuple[tuple[mx.array, int, int], ...]
    output_widths: tuple[int, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    neuron_len: int
    total_out: int
    total_tiles: int

    @classmethod
    def from_weights(
        cls,
        weights: tuple[PackedLinearWeight, ...],
    ) -> MetalLinearGroupWeight:
        if len(weights) < 2:
            raise ValueError("grouped linear requires at least two weights")
        neuron_len = int(weights[0].neuron_len)
        if neuron_len <= 0 or any(int(weight.neuron_len) != neuron_len for weight in weights):
            raise ValueError("grouped linear weights must share one input width")

        descriptors = np.zeros(
            (len(weights), _DESCRIPTOR_SIZE),
            dtype=np.int32,
        )
        streams: dict[str, list[mx.array]] = {
            "nint_q": [],
            "nint_sub_scale": [],
            "nint_sub_min": [],
            "nint_anchor_scale": [],
            "nint_anchor_min": [],
            "q8_q": [],
            "q8_scales": [],
            "vq_indices": [],
            "vq_state": [],
            "vq_aux": [],
            "vq_anchors": [],
            "vq_codebooks": [],
            "vq_scales": [],
            "vq_state_to_codebank": [],
            "vq_banks": [],
            "vq_parameters": [],
            "tpq_i4_packed": [],
            "tpq_i4_scales": [],
            "tpq_pq_indices8": [],
            "tpq_pq_indices16": [],
            "tpq_pq_indices_packed": [],
            "tpq_pq_codebooks": [],
        }
        offsets = {name: 0 for name in streams}
        rotation_variants: dict[tuple[int, int], int] = {}
        rotation_specs: list[tuple[mx.array, int, int]] = []
        output_widths: list[int] = []
        output_shapes: list[tuple[int, ...]] = []

        for projection, weight in enumerate(weights):
            descriptor = descriptors[projection]
            descriptor[_LOCAL_EXPERT] = 0
            descriptor[_OUT] = int(weight.out)
            descriptor[_K] = neuron_len
            output_widths.append(int(weight.out))
            output_shapes.append(
                tuple(map(int, weight.output_shape))
                if isinstance(weight, MetalVqWeight)
                else (int(weight.out),)
            )

            if isinstance(weight, MetalNintWeight):
                descriptor[_FAMILY] = _FAMILY_NINT
                descriptor[_NINT_BITS] = weight.bits
                descriptor[_NINT_GS] = weight.groupsize
                descriptor[_NINT_NG] = weight.groups
                descriptor[_NINT_Q_OFFSET] = offsets["nint_q"]
                descriptor[_NINT_SUB_OFFSET] = offsets["nint_sub_scale"]
                descriptor[_NINT_ANCHOR_OFFSET] = offsets["nint_anchor_scale"]
                descriptor[_NINT_Q5_EXEC] = int(weight.q5_exec)
                streams["nint_q"].append(weight.q_packed)
                streams["nint_sub_scale"].append(weight.sub_scale)
                streams["nint_sub_min"].append(weight.sub_min)
                streams["nint_anchor_scale"].append(weight.neuron_scale)
                streams["nint_anchor_min"].append(weight.neuron_min)
                offsets["nint_q"] += _size(weight.q_packed) + 2
                offsets["nint_sub_scale"] += _size(weight.sub_scale)
                offsets["nint_sub_min"] += _size(weight.sub_min)
                offsets["nint_anchor_scale"] += _size(weight.neuron_scale)
                offsets["nint_anchor_min"] += _size(weight.neuron_min)
                continue

            if isinstance(weight, MetalNint8ZeroWeight):
                descriptor[_FAMILY] = _FAMILY_NINT8_ZERO
                descriptor[_Q8_NG] = weight.groups
                descriptor[_Q8_Q_OFFSET] = offsets["q8_q"]
                descriptor[_Q8_SCALE_OFFSET] = offsets["q8_scales"]
                streams["q8_q"].append(weight.q)
                streams["q8_scales"].append(weight.scales)
                offsets["q8_q"] += _size(weight.q)
                offsets["q8_scales"] += _size(weight.scales)
                continue

            if isinstance(weight, MetalVqWeight):
                descriptor[_FAMILY] = _FAMILY_VQ
                descriptor[_VQ_GS] = weight.groupsize
                descriptor[_VQ_NG] = weight.groups
                descriptor[_VQ_VECTOR_SIZE] = weight.vector_size
                descriptor[_VQ_NVEC] = weight.vectors
                descriptor[_VQ_INDEX_BITS] = weight.index_bits
                descriptor[_VQ_STATE_BITS] = weight.state_bits
                descriptor[_VQ_STATES] = weight.states
                descriptor[_VQ_ENTRIES] = weight.entries
                descriptor[_VQ_CODE_BANKS] = weight.code_banks
                descriptor[_VQ_AUX_MODE] = weight.aux_mode
                descriptor[_VQ_CODE_BANK_MODE] = weight.code_bank_mode
                descriptor[_VQ_HAS_TABLE_BANKS] = int(weight.table_banks > 1)
                descriptor[_VQ_GROUPS_PER_SUPER] = weight.groups_per_super
                descriptor[_VQ_NSUPER] = weight.supergroups
                descriptor[_VQ_INDICES_OFFSET] = offsets["vq_indices"]
                descriptor[_VQ_STATE_OFFSET] = offsets["vq_state"]
                descriptor[_VQ_AUX_OFFSET] = offsets["vq_aux"]
                descriptor[_VQ_ANCHOR_OFFSET] = offsets["vq_anchors"]
                descriptor[_VQ_CODEBOOK_OFFSET] = offsets["vq_codebooks"]
                descriptor[_VQ_SCALE_OFFSET] = offsets["vq_scales"]
                descriptor[_VQ_STATE_BANK_OFFSET] = offsets["vq_state_to_codebank"]
                descriptor[_VQ_BANK_OFFSET] = offsets["vq_banks"]
                descriptor[_VQ_PARAMETER_OFFSET] = offsets["vq_parameters"]
                if weight.rotation_block:
                    key = (weight.rotation_block, weight.rotation_seed)
                    variant = rotation_variants.get(key, 0)
                    if variant == 0:
                        variant = len(rotation_specs) + 1
                        rotation_variants[key] = variant
                        rotation_specs.append(
                            (
                                weight.rotation_signs,
                                weight.rotation_block,
                                weight.rotation_seed,
                            )
                        )
                    descriptor[_VQ_ROTATION_VARIANT] = variant
                streams["vq_indices"].append(weight.indices_packed)
                streams["vq_state"].append(weight.state_packed)
                streams["vq_aux"].append(weight.aux_packed)
                streams["vq_anchors"].append(weight.anchors)
                streams["vq_codebooks"].append(weight.codebooks)
                streams["vq_scales"].append(weight.scale_lut)
                streams["vq_state_to_codebank"].append(weight.state_to_codebank)
                streams["vq_banks"].append(weight.bank_ids)
                streams["vq_parameters"].append(weight.parameters)
                offsets["vq_indices"] += _size(weight.indices_packed) + 2
                offsets["vq_state"] += _size(weight.state_packed) + 2
                offsets["vq_aux"] += _size(weight.aux_packed) + 2
                offsets["vq_anchors"] += _size(weight.anchors)
                offsets["vq_codebooks"] += _size(weight.codebooks)
                offsets["vq_scales"] += _size(weight.scale_lut)
                offsets["vq_state_to_codebank"] += _size(weight.state_to_codebank)
                offsets["vq_banks"] += _size(weight.bank_ids)
                offsets["vq_parameters"] += _size(weight.parameters)
                continue

            if isinstance(weight, MetalTpqInt4Weight):
                descriptor[_FAMILY] = _FAMILY_TPQ_INT4
                descriptor[_TPQ_I4_GROUP_SIZE] = weight.group_size
                descriptor[_TPQ_I4_GROUPS] = weight.groups
                descriptor[_TPQ_I4_PACKED_OFFSET] = offsets["tpq_i4_packed"]
                descriptor[_TPQ_I4_SCALE_OFFSET] = offsets["tpq_i4_scales"]
                streams["tpq_i4_packed"].append(weight.packed)
                streams["tpq_i4_scales"].append(weight.scales)
                offsets["tpq_i4_packed"] += _size(weight.packed)
                offsets["tpq_i4_scales"] += _size(weight.scales)
                continue

            if not isinstance(weight, MetalTpqPqWeight):
                raise TypeError(f"unsupported grouped linear weight {type(weight).__name__}")
            descriptor[_FAMILY] = _FAMILY_TPQ_PQ
            descriptor[_TPQ_PQ_BITS] = weight.index_bits
            descriptor[_TPQ_PQ_VECTOR_SIZE] = weight.vector_size
            descriptor[_TPQ_PQ_BLOCKS] = weight.blocks
            descriptor[_TPQ_PQ_CODEBOOK_OFFSET] = offsets["tpq_pq_codebooks"]
            bits = int(weight.index_bits)
            if bits == 8:
                index_stream = "tpq_pq_indices8"
            elif bits == 16:
                index_stream = "tpq_pq_indices16"
            else:
                index_stream = "tpq_pq_indices_packed"
            descriptor[_TPQ_PQ_INDEX_OFFSET] = offsets[index_stream]
            streams[index_stream].append(weight.indices)
            streams["tpq_pq_codebooks"].append(weight.codebook)
            offsets[index_stream] += _size(weight.indices)
            offsets["tpq_pq_codebooks"] += _size(weight.codebook)

        widths = tuple(output_widths)
        output_offsets = np.zeros((len(widths) + 1,), dtype=np.int32)
        output_offsets[1:] = np.cumsum(widths, dtype=np.int64)
        tile_counts = np.asarray(
            [(width + 7) // 8 for width in widths],
            dtype=np.int32,
        )
        tile_offsets = np.zeros((len(widths) + 1,), dtype=np.int32)
        tile_offsets[1:] = np.cumsum(tile_counts, dtype=np.int64)

        return cls(
            descriptors=mx.array(descriptors),
            projection_tile_offsets=mx.array(tile_offsets),
            projection_output_offsets=mx.array(output_offsets),
            nint_q=_join(streams["nint_q"], dtype=mx.uint8, padding=2),
            nint_sub_scale=_join(
                streams["nint_sub_scale"],
                dtype=mx.uint8,
            ),
            nint_sub_min=_join(streams["nint_sub_min"], dtype=mx.uint8),
            nint_anchor_scale=_join(
                streams["nint_anchor_scale"],
                dtype=mx.float32,
            ),
            nint_anchor_min=_join(
                streams["nint_anchor_min"],
                dtype=mx.float32,
            ),
            q8_q=_join(streams["q8_q"], dtype=mx.int8),
            q8_scales=_join(streams["q8_scales"], dtype=mx.float16),
            vq_indices=_join(
                streams["vq_indices"],
                dtype=mx.uint8,
                padding=2,
            ),
            vq_state=_join(
                streams["vq_state"],
                dtype=mx.uint8,
                padding=2,
            ),
            vq_aux=_join(
                streams["vq_aux"],
                dtype=mx.uint8,
                padding=2,
            ),
            vq_anchors=_join(streams["vq_anchors"], dtype=mx.float32),
            vq_codebooks=_join(streams["vq_codebooks"], dtype=mx.int8),
            vq_scales=_join(streams["vq_scales"], dtype=mx.float32),
            vq_state_to_codebank=_join(
                streams["vq_state_to_codebank"],
                dtype=mx.uint8,
            ),
            vq_banks=_join(streams["vq_banks"], dtype=mx.uint8),
            vq_parameters=_join(
                streams["vq_parameters"],
                dtype=mx.float32,
            ),
            tpq_i4_packed=_join(
                streams["tpq_i4_packed"],
                dtype=mx.uint8,
            ),
            tpq_i4_scales=_join(
                streams["tpq_i4_scales"],
                dtype=mx.float16,
            ),
            tpq_pq_indices8=_join(
                streams["tpq_pq_indices8"],
                dtype=mx.uint8,
            ),
            tpq_pq_indices16=_join(
                streams["tpq_pq_indices16"],
                dtype=mx.uint16,
            ),
            tpq_pq_indices_packed=_join(
                streams["tpq_pq_indices_packed"],
                dtype=mx.uint8,
            ),
            tpq_pq_codebooks=_join(
                streams["tpq_pq_codebooks"],
                dtype=mx.float16,
            ),
            descriptor_values=descriptors,
            rotation_specs=tuple(rotation_specs),
            output_widths=widths,
            output_shapes=tuple(output_shapes),
            neuron_len=neuron_len,
            total_out=int(output_offsets[-1]),
            total_tiles=int(tile_offsets[-1]),
        )

    @property
    def projections(self) -> int:
        return len(self.output_widths)

    @property
    def packed_nbytes(self) -> int:
        arrays = (
            self.descriptors,
            self.projection_tile_offsets,
            self.projection_output_offsets,
            self.nint_q,
            self.nint_sub_scale,
            self.nint_sub_min,
            self.nint_anchor_scale,
            self.nint_anchor_min,
            self.q8_q,
            self.q8_scales,
            self.vq_indices,
            self.vq_state,
            self.vq_aux,
            self.vq_anchors,
            self.vq_codebooks,
            self.vq_scales,
            self.vq_state_to_codebank,
            self.vq_banks,
            self.vq_parameters,
            self.tpq_i4_packed,
            self.tpq_i4_scales,
            self.tpq_pq_indices8,
            self.tpq_pq_indices16,
            self.tpq_pq_indices_packed,
            self.tpq_pq_codebooks,
        )
        return sum(int(array.nbytes) for array in arrays)


def grouped_linear_matmul(
    weight: MetalLinearGroupWeight,
    x: mx.array | np.ndarray,
) -> tuple[mx.array, ...]:
    """Apply all packed projections to a shared input in one Metal dispatch."""

    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError("grouped linear input must end in the shared weight width")
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    source = mx.contiguous(source)
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    if rows == 0:
        return tuple(
            mx.zeros((*prefix, *shape), dtype=source.dtype) for shape in weight.output_shapes
        )
    flattened = source.reshape((rows, weight.neuron_len))
    variants = [flattened]
    variants.extend(
        signed_hadamard(flattened, signs, block) for signs, block, _ in weight.rotation_specs
    )
    execution_input = mx.contiguous(mx.concatenate(variants, axis=0))
    result = _GROUPED_LINEAR_KERNEL(
        inputs=[
            weight.descriptors,
            weight.projection_tile_offsets,
            weight.projection_output_offsets,
            weight.nint_q,
            weight.nint_sub_scale,
            weight.nint_sub_min,
            weight.nint_anchor_scale,
            weight.nint_anchor_min,
            weight.q8_q,
            weight.q8_scales,
            weight.vq_indices,
            weight.vq_state,
            weight.vq_aux,
            weight.vq_anchors,
            weight.vq_codebooks,
            weight.vq_scales,
            weight.vq_state_to_codebank,
            weight.vq_banks,
            weight.vq_parameters,
            weight.tpq_i4_packed,
            weight.tpq_i4_scales,
            weight.tpq_pq_indices8,
            weight.tpq_pq_indices16,
            weight.tpq_pq_indices_packed,
            weight.tpq_pq_codebooks,
            execution_input,
        ],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("PROJECTIONS", weight.projections),
            ("K", weight.neuron_len),
            ("TOTAL_OUT", weight.total_out),
            ("TOTAL_TILES", weight.total_tiles),
            ("DESCRIPTOR_SIZE", _DESCRIPTOR_SIZE),
        ],
        grid=(rows * weight.total_tiles * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, weight.total_out)],
        output_dtypes=[source.dtype],
    )[0]
    outputs: list[mx.array] = []
    offset = 0
    for width, shape in zip(
        weight.output_widths,
        weight.output_shapes,
        strict=True,
    ):
        outputs.append(result[:, offset : offset + width].reshape((*prefix, *shape)))
        offset += width
    return tuple(outputs)


__all__ = [
    "MetalLinearGroupWeight",
    "PackedLinearWeight",
    "grouped_linear_matmul",
]
