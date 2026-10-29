"""Metal projections over the shared RTRELL05 TQ payload."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np

from mfq.formats.tq import RELATIVE, TQTensor


@lru_cache(None)
def _kernel(pair=False):
    header = Path(__file__).with_name("tq_decode.h").read_text()
    source = r"""
uint row = thread_position_in_grid.x / LANES;
uint lane = thread_position_in_grid.x % LANES;
uint route = thread_position_in_grid.y;
if (row >= OUT) return;
int expert = ids[route];
uint input_row = SHARED_INPUT ? route / ROUTES : route;
float gate = 0.f, up = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint weight_row = uint(expert) * OUT + row;
    if (PAIR) {
        float2 result = tq_dot_pair<T, LANES, COLUMN_G, COLUMN_U, WIDTH_G, WIDTH_U, K, GS_G, GS_U>(
            words_g, descriptors_g[weight_row], scales_g, words_u, descriptors_u[weight_row], scales_u,
            relative, weight_row, weight_row, x + input_row * K, lane);
        gate = result.x; up = result.y;
    } else gate = tq_dot<T, LANES, COLUMN_G, WIDTH_G, K, GS_G>(
        words_g, descriptors_g[weight_row], scales_g, relative, weight_row, x + input_row * K, lane);
}
for (uint step = LANES / 2; step; step >>= 1) {
    gate += simd_shuffle_down(gate, step);
    if (PAIR) up += simd_shuffle_down(up, step);
}
if (lane == 0) {
    if (PAIR) {
        gate = float(T(gate)); up = float(T(up));
        float limit = params[0];
        if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
        gate = GELU ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
                    : gate / (1.f + exp(-gate));
        gate *= up;
    }
    y[route * OUT + row] = T(gate);
}
"""
    return mx.fast.metal_kernel(
        name="mfq_tq_pair" if pair else "mfq_tq_project",
        input_names=["words_g", "descriptors_g", "scales_g", "relative",
                     "words_u", "descriptors_u", "scales_u", "x", "ids", "params"],
        output_names=["y"], header=header, source=source,
    )


class Projection:
    def __init__(self, tensor, *, input_columns=None, experts=1):
        if not isinstance(tensor, TQTensor):
            raise TypeError("expected a TQTensor")
        if not isinstance(experts, int) or experts <= 0 or tensor.rows % experts:
            raise ValueError("expert count must divide the number of TQ rows")
        self.input_columns = tensor.width if input_columns is None else input_columns
        if not isinstance(self.input_columns, int) or not 0 < self.input_columns <= tensor.width:
            raise ValueError("invalid logical input width")
        self.tensor = tensor
        self.experts = experts
        self.output_size = tensor.rows // experts
        self.words = mx.array(tensor.packed.view("<u4"))
        self.descriptors = mx.array(tensor.descriptors)
        if not tensor.columns and tensor.gs >= 256:
            codes = np.empty(tensor.scales.size * 2, np.uint8)
            codes[::2], codes[1::2] = tensor.scales & 15, tensor.scales >> 4
            gains = RELATIVE[codes[:tensor.rows * tensor.groups].reshape(tensor.rows, tensor.groups)]
            self.scales = mx.array((tensor.anchors.astype(np.float32)[:, None]
                                   * gains.astype(np.float32)).reshape(-1))
        else:
            self.scales = mx.array(tensor.scales)
        self.relative = mx.array(RELATIVE)
        mx.eval(self.words, self.descriptors, self.scales, self.relative)

    def __call__(self, x, ids=None):
        return self._run(x, ids)

    def routed_reduce(self, x, ids, weights):
        if (x.ndim == 0 or x.shape[-1] != self.input_columns or ids.ndim != 2
                or not ids.size or ids.dtype not in (mx.int32, mx.uint32) or weights.shape != ids.shape):
            raise ValueError("TQ routed reduction geometry differs")
        tokens, routes = ids.shape
        source_rows = x.size // self.input_columns
        if source_rows not in (tokens, tokens * routes) or x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("TQ routed reduction activations differ")
        if self.input_columns >= 1024 or not 2 <= routes <= 32:
            return mx.sum(self(x, ids).astype(mx.float32) * weights.astype(mx.float32)[:, :, None], axis=1).astype(x.dtype)
        lanes = 16
        return _routed_reduce_kernel()(
            inputs=[self.words, self.descriptors, self.scales, self.relative,
                    mx.contiguous(x.reshape(-1, self.input_columns)), mx.contiguous(ids),
                    mx.contiguous(weights.astype(mx.float32))],
            template=[("T", x.dtype), ("LANES", lanes), ("OUT", self.output_size),
                      ("K", self.input_columns), ("WIDTH", self.tensor.width),
                      ("GS", self.tensor.gs), ("COLUMNS", self.tensor.columns),
                      ("EXPERTS", self.experts), ("ROUTES", routes), ("SHARED_INPUT", source_rows == tokens)],
            grid=(self.output_size * routes * lanes, tokens, 1), threadgroup=(routes * lanes, 1, 1),
            output_shapes=[(tokens, self.output_size)], output_dtypes=[x.dtype],
        )[0]

    def swiglu(self, up, x, ids=None, *, limit=0., gelu=False):
        if (self.input_columns, self.experts, self.output_size) != (
            up.input_columns, up.experts, up.output_size
        ):
            raise ValueError("Gate/Up TQ geometry differs")
        return self._run(x, ids, up=up, limit=limit, gelu=gelu)

    def _run(self, x, ids, up=None, limit=0., gelu=False, lanes=None):
        if x.ndim == 0 or x.shape[-1] != self.input_columns:
            raise ValueError("TQ activation width differs")
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("TQ requires floating activations")
        if ids is None:
            if self.experts != 1:
                raise ValueError("routed TQ requires expert IDs")
            tokens, routes = x.size // self.input_columns, 1
            ids = mx.zeros((tokens, 1), mx.int32)
            output_shape = (*x.shape[:-1], self.output_size)
        else:
            if ids.ndim != 2 or not ids.size or ids.dtype not in (mx.int32, mx.uint32):
                raise ValueError("expert IDs must be a nonempty 2D integer array")
            tokens, routes = ids.shape
            output_shape = (tokens, routes, self.output_size)
        source_rows = x.size // self.input_columns
        if source_rows not in (tokens, tokens * routes):
            raise ValueError("TQ input rows differ from routed IDs")
        other = self if up is None else up
        if lanes is None:
            lanes = 16 if up is None or self.input_columns < 1024 else 32
        rows_per_group = 128 // lanes
        total = tokens * routes
        result = _kernel(up is not None)(
            inputs=[self.words, self.descriptors, self.scales, self.relative,
                    other.words, other.descriptors, other.scales,
                    mx.contiguous(x.reshape(-1, self.input_columns)), mx.contiguous(ids),
                    mx.array([limit], mx.float32)],
            template=[("T", x.dtype), ("LANES", lanes), ("OUT", self.output_size),
                      ("K", self.input_columns), ("WIDTH_G", self.tensor.width),
                      ("WIDTH_U", other.tensor.width), ("GS_G", self.tensor.gs),
                      ("GS_U", other.tensor.gs), ("COLUMN_G", self.tensor.columns),
                      ("COLUMN_U", other.tensor.columns), ("EXPERTS", self.experts),
                      ("ROUTES", routes), ("SHARED_INPUT", source_rows == tokens),
                      ("PAIR", up is not None), ("GELU", gelu)],
            grid=(((self.output_size + rows_per_group - 1) // rows_per_group) * 128, total, 1),
            threadgroup=(128, 1, 1), output_shapes=[(total, self.output_size)],
            output_dtypes=[x.dtype],
        )[0]
        return result.reshape(output_shape)


@lru_cache(None)
def _routed_reduce_kernel():
    return mx.fast.metal_kernel(
        name="mfq_tq_routed_reduce", input_names=["words", "descriptors", "scales", "relative", "x", "ids", "weights"],
        output_names=["y"], header=Path(__file__).with_name("tq_decode.h").read_text(), source=r"""
threadgroup T partial[ROUTES];
uint row = threadgroup_position_in_grid.x;
uint local = thread_position_in_threadgroup.x;
uint lane = local % LANES, route = local / LANES;
uint token = threadgroup_position_in_grid.y;
int expert = ids[token * ROUTES + route];
float value = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint weight_row = uint(expert) * OUT + row;
    value = tq_dot<T, LANES, COLUMNS, WIDTH, K, GS>(words, descriptors[weight_row],
        scales, relative, weight_row, x + (SHARED_INPUT ? token : token * ROUTES + route) * K, lane);
}
for (uint step = LANES / 2; step; step >>= 1) value += simd_shuffle_down(value, step);
if (lane == 0) partial[route] = T(value);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local == 0) {
    float total = 0.f;
    for (uint r = 0; r < ROUTES; ++r) total = fma(float(partial[r]), weights[token * ROUTES + r], total);
    y[token * OUT + row] = T(total);
}
""",
    )
