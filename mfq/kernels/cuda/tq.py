"""Fixed-book, row-dynamic Trellis operators for ordinary 2D projections.

FP16 Gaussian book, FP32 accumulation, FP16/BF16/FP32 activations.
FP16 projections apply shared scales after partial dots, avoiding per-weight
FP16 rounding. BF16/FP32 projections retain FP16 weight reconstruction.
Supplying outputs/workspace makes calls allocation-free and CUDA-graph safe.
"""
import ctypes as ct
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from mfq.formats.tq import RELATIVE, TQTensor


class ProjectionView(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in
                ('words', 'descriptors', 'scales', 'relative', 'bias')] + [
                    (name, ct.c_int) for name in
                    ('rows', 'stored_columns', 'input_columns', 'gslog', 'columns')]


class ProjectionJob(ct.Structure):
    _fields_ = [('weight', ProjectionView), ('output_offset', ct.c_int64),
                ('block_end', ct.c_int)]


class IndexedProjectionJob(ct.Structure):
    _fields_ = [('weight', ProjectionView), ('second', ProjectionView),
                ('input_offsets', ct.c_int64 * 4), ('output_offsets', ct.c_int64 * 4),
                ('count', ct.c_int)]


class IndexedWideProjectionJob(ct.Structure):
    _fields_ = [('weight', ProjectionView), ('input_offsets', ct.c_int64 * 8),
                ('output_offsets', ct.c_int64 * 8), ('count', ct.c_int)]


class IndexedWidePairJob(ct.Structure):
    _fields_ = [('weight', ProjectionView), ('second', ProjectionView),
                ('input_offsets', ct.c_int64 * 8), ('output_offsets', ct.c_int64 * 8),
                ('count', ct.c_int)]


class PrefillMoeJob(ct.Structure):
    _fields_ = [('gate', ProjectionView), ('up', ProjectionView),
                ('down', ProjectionView), ('first', ct.c_int64), ('count', ct.c_int)]


@lru_cache(None)
def library():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    path = load(
        name="mfq_tq_ops",
        sources=[str(directory / "tq_ops.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        extra_ldflags=["cublas.lib"] if os.name == "nt" else ["-lcublas"],
        is_python_module=False,
    )
    lib = ct.CDLL(path)
    P, I, V = ct.c_void_p, ct.c_int, ProjectionView
    lib.mfq_tq_gemv.argtypes = [V, P, P, I, I, I, P]
    lib.mfq_tq_swiglu.argtypes = [V, V, P, P, I, I, I, P]
    lib.mfq_tq_ffn.argtypes = [V, V, V, P, P, P, I, P]
    lib.mfq_tq_parallel.argtypes = [P, I, I, P, P, I, I, I, I, P]
    lib.mfq_tq_parallel_tokens.argtypes = [P, I, I, P, P, I, I, I, P]
    lib.mfq_tq_q8_tokens.argtypes = [V, P, P, I, I, I, P, ct.c_uint64, P]
    lib.mfq_tq_q8_parallel_tokens.argtypes = [P, P, I, I, P, P, I, I, I, P, ct.c_uint64, I, I, P]
    lib.mfq_tq_float_gemm.argtypes = [V, P, P, I, I, I, P, ct.c_uint64, P, P]
    lib.mfq_tq_gemm.argtypes = [V, P, P, I, P, ct.c_uint64, P, P]
    lib.mfq_tq_grouped_gemm.argtypes = [V, P, P, I, I, P, ct.c_uint64, P, P]
    lib.mfq_tq_grouped_tokens.argtypes = [V, P, P, I, I, I, P]
    lib.mfq_tq_gemv_tokens.argtypes = [V, P, P, I, I, P]
    lib.mfq_tq_embedding_rows.argtypes = [P, I, I, P, I, P]
    lib.mfq_tq_routed_swiglu.argtypes = [P, P, P, P, P, I, I, I, ct.c_float, P]
    lib.mfq_tq_routed_down.argtypes = [P, P, P, P, I, I, I, I, P]
    lib.mfq_tq_indexed.argtypes = [P, P, P, I, I, I, I, P]
    lib.mfq_tq_indexed_wide.argtypes = [P, P, P, I, I, I, I, P]
    lib.mfq_tq_indexed_swiglu.argtypes = [P, P, P, I, I, I, I, ct.c_float, P]
    lib.mfq_tq_indexed_wide_swiglu.argtypes = [P, P, P, I, I, I, I, ct.c_float, P]
    lib.mfq_tq_weighted_reduce.argtypes = [P, P, P, I, I, I, I, P]
    lib.mfq_tq_prefill_moe.argtypes = [P, P, P, P, I, I, I, I, I, ct.c_float, P, ct.c_uint64, P]
    lib.mfq_tq_prefill_moe_gemm.argtypes = [P, P, P, P, I, I, I, I, I, ct.c_float,
                                         P, ct.c_uint64, P, ct.c_uint64, P, P]
    for name in ('mfq_tq_gemv', 'mfq_tq_swiglu', 'mfq_tq_ffn', 'mfq_tq_parallel',
                 'mfq_tq_parallel_tokens', 'mfq_tq_q8_tokens', 'mfq_tq_q8_parallel_tokens',
                 'mfq_tq_float_gemm', 'mfq_tq_gemm',
                 'mfq_tq_grouped_gemm', 'mfq_tq_grouped_tokens', 'mfq_tq_gemv_tokens',
                 'mfq_tq_embedding_rows', 'mfq_tq_routed_swiglu',
                 'mfq_tq_routed_down', 'mfq_tq_indexed', 'mfq_tq_indexed_wide', 'mfq_tq_indexed_swiglu',
                 'mfq_tq_indexed_wide_swiglu', 'mfq_tq_weighted_reduce', 'mfq_tq_prefill_moe',
                 'mfq_tq_prefill_moe_gemm'):
        getattr(lib, name).restype = I
    return lib


def _status(code):
    if code:
        raise RuntimeError(f'Trellis operator returned {code}')


def _dtype(dtype):
    return {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype]


def _vector(x, length, device, dtype=None):
    if (x.device != device or x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            x.ndim != 1 or x.numel() != length or not x.is_contiguous() or
            (dtype is not None and x.dtype != dtype)):
        raise ValueError(f'Expected contiguous {dtype or "FP16/BF16/FP32"} vector of length {length} on {device}')


def _overlap(x, y):
    return max(x.data_ptr(), y.data_ptr()) < min(
        x.data_ptr() + x.numel() * x.element_size(),
        y.data_ptr() + y.numel() * y.element_size())


def weighted_reduce(expert_output, weights, out=None):
    """Return the route-weighted MoE output, with FP32 accumulation."""
    if (expert_output.device.type != 'cuda' or expert_output.ndim != 3 or
            expert_output.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            not expert_output.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP16/BF16/FP32 [tokens, routes, width]')
    tokens, routes, width = expert_output.shape
    if (routes < 1 or width < 1 or routes > 2**31 - 1 or width > 2**31 - 1 or
            weights.device != expert_output.device or weights.dtype != torch.float32 or
            weights.shape != (tokens, routes) or not weights.is_contiguous()):
        raise ValueError('Expected matching contiguous CUDA FP32 routing weights')
    if out is None:
        out = torch.empty((tokens, width), dtype=expert_output.dtype, device=expert_output.device)
    if (out.shape != (tokens, width) or out.device != expert_output.device or
            out.dtype != expert_output.dtype or not out.is_contiguous() or
            _overlap(out, expert_output) or _overlap(out, weights)):
        raise ValueError('Output must match [tokens, width] and not overlap the inputs')
    element = expert_output.element_size()
    with torch.cuda.device(expert_output.device):
        lib = library()
        stream = torch.cuda.current_stream(expert_output.device).cuda_stream
        for start in range(0, tokens, 65535):
            _status(lib.mfq_tq_weighted_reduce(
                expert_output.data_ptr() + start * routes * width * element,
                weights.data_ptr() + start * routes * weights.element_size(),
                out.data_ptr() + start * width * element,
                min(65535, tokens - start), routes, width, _dtype(expert_output.dtype), stream))
    return out


class Projection:
    _library = staticmethod(library)

    def __init__(self, data, input_columns=None, bias=None, device='cuda'):
        if isinstance(data, (str, Path)):
            data = TQTensor.load(data)
        if not isinstance(data, TQTensor):
            raise TypeError('Expected a TQTensor or RTRELL05 path')
        data = data.components()
        self.rows, self.stored_columns = int(data['rows']), int(data['width'])
        self.input_columns = self.stored_columns if input_columns is None else int(input_columns)
        gs, columns = int(data['gs']), bool(data['columns'])
        if (self.rows <= 0 or self.stored_columns % 256 or
                not 0 < self.input_columns <= self.stored_columns or
                (not columns and (gs < 16 or gs > 2**20 or gs & (gs - 1)))):
            raise ValueError('Invalid projection geometry or subgroup size')
        target = torch.device(device)
        if target.type != 'cuda':
            raise ValueError('Trellis CUDA operators require a CUDA device')
        with torch.cuda.device(target):
            self.words = torch.as_tensor(np.array(data['packed'], copy=True), device=target)
            self.descriptors = torch.as_tensor(np.array(data['descriptors'], copy=True), device=target)
            self.scales = torch.as_tensor(np.array(data['scales'], copy=True), device=target)
            self.relative = torch.tensor(RELATIVE.copy(), device=target)
            self.device = self.words.device
            self.bias = None if bias is None else torch.as_tensor(
                bias, dtype=torch.float32, device=self.device).contiguous()
        if self.bias is not None and self.bias.shape != (self.rows,):
            raise ValueError('Bias length must equal projection output rows')
        if self.descriptors.numel() != self.rows:
            raise ValueError('Row descriptor count differs')
        expected_scales = self.stored_columns if columns else (
            self.rows * ((self.stored_columns + gs - 1) // gs) + 1) // 2
        if self.scales.numel() != expected_scales:
            raise ValueError('Scale payload length differs')
        self.view = ProjectionView(self.words.data_ptr(), self.descriptors.data_ptr(),
            self.scales.data_ptr(), self.relative.data_ptr(),
            self.bias.data_ptr() if self.bias is not None else None,
            self.rows, self.stored_columns, self.input_columns,
            0 if columns else gs.bit_length() - 1, int(columns))
        with torch.cuda.device(self.device):
            self.lib = self._library()

    def gemv(self, x, out=None, *, warps=0, rows_per_block=0):
        _vector(x, self.input_columns, self.device)
        if out is None:
            out = torch.empty(self.rows, dtype=x.dtype, device=self.device)
        _vector(out, self.rows, self.device, x.dtype)
        if _overlap(x, out):
            raise ValueError('GEMV input and output must not overlap')
        with torch.cuda.device(self.device):
            _status(self.lib.mfq_tq_gemv(self.view, x.data_ptr(), out.data_ptr(),
                _dtype(x.dtype), warps, rows_per_block,
                torch.cuda.current_stream(self.device).cuda_stream))
        return out


class SwiGLU:
    def __init__(self, gate, up):
        if (gate.rows != up.rows or gate.input_columns != up.input_columns or
                gate.device != up.device):
            raise ValueError('Gate/Up shapes, devices and operator libraries must match')
        self.gate, self.up = gate, up

    def __call__(self, x, out=None, *, warps=0, rows_per_block=0):
        g, u = self.gate, self.up
        _vector(x, g.input_columns, g.device)
        if out is None:
            out = torch.empty(g.rows, dtype=x.dtype, device=g.device)
        _vector(out, g.rows, g.device, x.dtype)
        if _overlap(x, out):
            raise ValueError('SwiGLU input and output must not overlap')
        with torch.cuda.device(g.device):
            _status(g.lib.mfq_tq_swiglu(g.view, u.view, x.data_ptr(), out.data_ptr(),
                _dtype(x.dtype), warps, rows_per_block,
                torch.cuda.current_stream(g.device).cuda_stream))
        return out


class ParallelProjections:
    """Fuse independent row projections without copying their weight payloads."""
    def __init__(self, projections):
        self.projections = tuple(projections)
        if not self.projections:
            raise ValueError('At least one projection is required')
        first = self.projections[0]
        self.device, self.input_columns, self.lib = first.device, first.input_columns, first.lib
        jobs, blocks, offset = [], 0, 0
        for op in self.projections:
            if op.device != self.device or op.input_columns != self.input_columns:
                raise ValueError('Parallel projections must share their input width and device')
            blocks += (op.rows + 3) // 4
            jobs.append(ProjectionJob(op.view, offset, blocks))
            offset += op.rows
        if blocks > 2**31 - 1 or offset > 2**31 - 1 or self.input_columns > 2**31 - 1:
            raise ValueError('Parallel projection plan exceeds CUDA index range')
        self.blocks, self.output_columns = blocks, offset
        self.host_plan = (ProjectionJob * len(jobs))(*jobs)
        packed = np.frombuffer(self.host_plan, dtype=np.uint8).copy()
        with torch.cuda.device(self.device):
            self.plan = torch.tensor(packed, device=self.device)

    def __call__(self, x, out=None):
        if (x.device != self.device or x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
                x.ndim < 1 or x.size(-1) != self.input_columns or not x.is_contiguous()):
            raise ValueError('Expected contiguous activations with the parallel input width and device')
        shape = (*x.shape[:-1], self.output_columns)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=self.device)
        if (out.shape != shape or out.device != self.device or out.dtype != x.dtype or
                not out.is_contiguous() or _overlap(x, out)):
            raise ValueError('Parallel output must match the combined shape and not overlap the input')
        tokens, element = x.numel() // self.input_columns, x.element_size()
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream
            for start in range(0, tokens, 65535):
                _status(self.lib.mfq_tq_parallel(self.plan.data_ptr(), len(self.projections), self.blocks,
                    x.data_ptr() + start * self.input_columns * element,
                    out.data_ptr() + start * self.output_columns * element,
                    min(65535, tokens - start), self.input_columns, self.output_columns, _dtype(x.dtype), stream))
        return out.split([op.rows for op in self.projections], dim=-1)


class SwiGLUFFN(SwiGLU):
    """Two CUDA kernels: fused Gate/Up/SwiGLU, then Down GEMV."""
    def __init__(self, gate, up, down):
        super().__init__(gate, up)
        if (down.input_columns != gate.rows or down.device != gate.device):
            raise ValueError('Down input width, device and operator library must match Gate/Up')
        self.down = down

    def __call__(self, x, out=None, *, workspace=None):
        g, u, d = self.gate, self.up, self.down
        _vector(x, g.input_columns, g.device)
        if out is None:
            out = torch.empty(d.rows, dtype=x.dtype, device=g.device)
        if workspace is None:
            workspace = torch.empty(g.rows, dtype=x.dtype, device=g.device)
        _vector(out, d.rows, g.device, x.dtype)
        _vector(workspace, g.rows, g.device, x.dtype)
        if _overlap(workspace, x) or _overlap(workspace, out):
            raise ValueError('FFN workspace must not overlap input/output')
        with torch.cuda.device(g.device):
            _status(g.lib.mfq_tq_ffn(g.view, u.view, d.view, x.data_ptr(),
                workspace.data_ptr(), out.data_ptr(), _dtype(x.dtype),
                torch.cuda.current_stream(g.device).cuda_stream))
        return out
