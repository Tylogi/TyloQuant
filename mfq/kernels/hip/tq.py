"""Packed TQ projections and fixed-weight input gradients on ROCm."""
from __future__ import annotations

import ctypes as ct
import hashlib
from functools import lru_cache
from pathlib import Path

import torch

from mfq.kernels.cuda.tq import Projection as _Projection, ProjectionView, _dtype, _overlap

_MM_FP32_OUTPUT = "out_dtype" in (torch.mm.__doc__ or "")


@lru_cache(None)
def library():
    from torch.utils.cpp_extension import load

    if not torch.version.hip:
        raise RuntimeError("TQ HIP kernels require a ROCm build of PyTorch")
    source = Path(__file__).with_name("tq_projection.hip")
    headers = b"".join(source.with_name(name).read_bytes() for name in
                       ("tq_projection.h", "tq_vector.h", "tq_float_tokens.h", "tq_materialize.h", "tq_fused.h"))
    path = load(name="mfq_tq_ops_hip", sources=[str(source)],
                extra_cuda_cflags=["-O3", "-ffp-contract=off",
                                  "-DMFQ_TQ_HEADERS_SHA=0x" + hashlib.sha256(headers).hexdigest()[:16]],
                is_python_module=False)
    lib = ct.CDLL(path)
    P, I = ct.c_void_p, ct.c_int
    for name in ("mfq_tq_forward", "mfq_tq_backward"):
        getattr(lib, name).argtypes = [ProjectionView, P, P, I, I, P]
        getattr(lib, name).restype = I
    lib.mfq_tq_gemv.argtypes = [ProjectionView, P, P, I, I, I, P]
    lib.mfq_tq_gemv.restype = I
    lib.mfq_tq_materialize.argtypes = [ProjectionView, P, I, P]
    lib.mfq_tq_materialize.restype = I
    lib.mfq_tq_materialize_bf16_pair.argtypes = [ProjectionView, P, P, P]
    lib.mfq_tq_materialize_bf16_pair.restype = I
    lib.mfq_tq_materialize_bf16.argtypes = [ProjectionView, P, I, P]
    lib.mfq_tq_materialize_bf16.restype = I
    return lib


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, projection):
        ctx.projection = projection
        return projection._run(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        return ctx.projection.backward(gradient.contiguous()), None


class Projection(_Projection):
    """A fixed compressed matrix; backward returns gradients for activations."""

    _library = staticmethod(library)

    def __init__(self, data, input_columns=None, bias=None, device="cuda", *, staging=True, split_bf16=None):
        super().__init__(data, input_columns=input_columns, bias=bias, device=device)
        self.staging = bool(staging)
        self.split_bf16 = split_bf16

    def _use_gemm(self, tokens, backward):
        if not tokens:
            return False
        if not backward and (tokens <= 8 or (tokens <= 32 and self.rows <= 64)):
            return False
        area = self.rows * self.input_columns
        reduction = self.rows if backward else self.input_columns
        # Tiny reductions are cheaper directly from the compressed weights.
        return (reduction > 64 and area >= 16384 and
                (reduction >= 512 or tokens * area >= 2**27))

    @torch.no_grad()
    def _gemm(self, x, out, *, backward):
        tokens = x.numel() // x.size(-1)
        work = tokens * self.rows * self.input_columns
        split = self.split_bf16 is True or (
            self.split_bf16 is None and tokens >= 128 and work >= 2**30)
        if split and x.dtype == torch.bfloat16 and _MM_FP32_OUTPUT:
            # Exact BF16 components expand the reduction dimension, allowing
            # one BF16 matrix-core GEMM with FP32 accumulation.
            shape = (self.rows * 2, self.input_columns) if backward else (self.rows, self.input_columns * 2)
            matrix = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            stream = torch.cuda.current_stream(self.device).cuda_stream
            code = self.lib.mfq_tq_materialize_bf16(self.view, matrix.data_ptr(), int(backward), stream)
            if code:
                raise RuntimeError(f"TQ HIP materialization returned {code}")
            activation = x.reshape(-1, x.size(-1))
            expanded = torch.cat((activation, activation), dim=1)
            result = torch.mm(expanded, matrix if backward else matrix.T, out_dtype=torch.float32)
            if not backward and self.bias is not None:
                result.add_(self.bias)
            out.copy_(result.reshape_as(out))
            return out
        # BF16 inputs use FP32 weights to retain the exact FP16 reconstruction.
        dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
        matrix = torch.empty((self.rows, self.input_columns), dtype=dtype, device=self.device)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        code = self.lib.mfq_tq_materialize(self.view, matrix.data_ptr(), _dtype(dtype), stream)
        if code:
            raise RuntimeError(f"TQ HIP materialization returned {code}")
        activation = x.reshape(-1, x.size(-1)).to(dtype)
        weights = matrix if backward else matrix.T
        if dtype == torch.float16 and _MM_FP32_OUTPUT:
            result = torch.mm(activation, weights, out_dtype=torch.float32)
        else:
            result = torch.mm(activation.float(), weights.float())
        if not backward and self.bias is not None:
            result.add_(self.bias)
        out.copy_(result.reshape_as(out))
        return out

    def _run(self, x, out=None, *, backward=False):
        width, result_width = ((self.rows, self.input_columns) if backward else
                               (self.input_columns, self.rows))
        if (x.device != self.device or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or x.ndim < 1 or x.size(-1) != width or not x.is_contiguous()):
            raise ValueError(f"Expected contiguous FP16/BF16/FP32 activations with width {width} on {self.device}")
        shape = (*x.shape[:-1], result_width)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=self.device)
        if (out.shape != shape or out.dtype != x.dtype or out.device != self.device
                or not out.is_contiguous() or _overlap(x, out)):
            raise ValueError("Output must match the result shape, dtype and device without input overlap")
        tokens = x.numel() // width
        function = self.lib.mfq_tq_backward if backward else self.lib.mfq_tq_forward
        with torch.cuda.device(self.device):
            force_split = self.split_bf16 is True and x.dtype == torch.bfloat16 and tokens > 0
            if self.staging and (force_split or self._use_gemm(tokens, backward)):
                return self._gemm(x, out, backward=backward)
            stream = torch.cuda.current_stream(self.device).cuda_stream
            for start in range(0, tokens, 65535 * 16):
                code = function(self.view, x.data_ptr() + start * width * x.element_size(),
                                out.data_ptr() + start * result_width * out.element_size(),
                                min(65535 * 16, tokens - start), _dtype(x.dtype), stream)
                if code:
                    raise RuntimeError(f"TQ HIP operator returned {code}")
        return out

    def __call__(self, x):
        return _Linear.apply(x, self)

    def gemv(self, x, out=None):
        if x.ndim != 1:
            raise ValueError("GEMV requires a one-dimensional activation vector")
        return self.forward(x, out)

    def forward(self, x, out=None):
        """Run decode or prefill; explicit output buffers do not track gradients."""
        if out is None:
            return self(x)
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("Use projection(x) when activation gradients are required")
        return self._run(x, out)

    def backward(self, gradient, out=None):
        """Compute dX = dY @ W directly from the compressed matrix."""
        return self._run(gradient, out, backward=True)


class SwiGLU:
    def __init__(self, gate, up):
        if (gate.rows != up.rows or gate.input_columns != up.input_columns or gate.device != up.device):
            raise ValueError("Gate and Up must share their geometry and device")
        self.gate, self.up = gate, up

    def __call__(self, x):
        return torch.nn.functional.silu(self.gate(x)) * self.up(x)


class SwiGLUFFN(SwiGLU):
    def __init__(self, gate, up, down):
        super().__init__(gate, up)
        if down.input_columns != gate.rows or down.device != gate.device:
            raise ValueError("Down must match the Gate/Up output width and device")
        self.down = down

    def __call__(self, x):
        return self.down(super().__call__(x))
