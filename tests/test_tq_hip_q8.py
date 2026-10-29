"""CUDA-equivalent Q8 activation path; separate from exact reconstruction."""
import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("ROCm unavailable", allow_module_level=True)

from mfq.formats.tq import RELATIVE, unpack_states
from mfq.kernels.cuda.tq import ProjectionView
from mfq.kernels.hip.tq import Projection
from tests.test_tq_hip import fixture


@pytest.fixture(scope="module")
def q8():
    from torch.utils.cpp_extension import load
    source = Path(__file__).parents[1] / "mfq/kernels/hip/tq_q8.hip"
    headers = source.with_name("tq_vector.h").read_bytes() + source.with_name("tq_projection.h").read_bytes()
    lib = ct.CDLL(load(name="mfq_tq_q8_test", sources=[str(source)],
        extra_cuda_cflags=["-O3", "-ffp-contract=off",
            "-DMFQ_TQ_HEADERS_SHA=0x" + hashlib.sha256(headers).hexdigest()[:16]], is_python_module=False))
    fun = lib.mfq_tq_q8_tokens
    fun.argtypes = [ProjectionView, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
        ct.c_void_p, ct.c_uint64, ct.c_void_p]
    fun.restype = ct.c_int
    fun.lib = lib
    return fun


def affine(tensor, width):
    result = np.empty((tensor.rows, tensor.width), dtype=np.float32)
    if not tensor.columns:
        selectors = np.stack((tensor.scales & 15, tensor.scales >> 4), 1).reshape(-1)
    for row, bits in enumerate(tensor.bits):
        offset = int(tensor.descriptors[row] & np.uint64(0xffffffff)) * 4
        n = tensor.width * int(bits) // 8
        states = unpack_states(tensor.packed[offset:offset+n].reshape(-1, 32*int(bits)), int(bits)).reshape(-1)
        code = states.astype(np.uint32) * np.uint32(0x83dcd12d)
        centered = sum(((code >> shift) & 255).astype(np.int32) for shift in (0, 8, 16, 24)) - 510
        scales = np.float32(tensor.anchors[row])
        if not tensor.columns:
            scales = scales * np.repeat(RELATIVE[selectors[row*tensor.groups:(row+1)*tensor.groups]], tensor.gs)[:tensor.width]
        result[row] = (centered.astype(np.float32)*np.float32(.00676727294921875) - np.float32(.0018157958984375)) * scales
    return torch.from_numpy(result[:, :width].copy()).to("cuda").double()


@pytest.mark.parametrize("columns,groupsize", [(False, 16), (False, 512), (True, 512)])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("dtype,code", [(torch.float16, 0), (torch.float32, 2)])
def test_q8_partial_groups_match_affine_oracle(q8, columns, groupsize, tokens, dtype, code):
    width = 273
    tensor = fixture(19, width, columns, groupsize)
    op = Projection(tensor, input_columns=width)
    x = torch.randn(tokens, width, device="cuda", dtype=dtype)
    scaled = x.float()
    if columns:
        scaled = scaled * torch.from_numpy(tensor.scales[:width].copy()).to("cuda").float()
    chunks = torch.nn.functional.pad(scaled, (0, (-width) % 16)).reshape(tokens, -1, 16)
    peak = chunks.abs().amax(-1, keepdim=True)
    scale = torch.where(peak > 0, peak/127, torch.ones_like(peak))
    values = ((chunks/scale).round().clamp(-127, 127)*scale).reshape(tokens, -1)[:, :width]
    expected = values.double() @ affine(tensor, width).T
    out = torch.empty(tokens, tensor.rows, device="cuda", dtype=dtype)
    scratch = torch.empty(tokens*32*24, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream
    assert q8(op.view, x.data_ptr(), out.data_ptr(), tokens, code, scratch.data_ptr(), scratch.numel(), stream) == 0
    rms = (out.double()-expected).square().mean().sqrt()/expected.square().mean().sqrt()
    assert rms < (.001 if code == 0 else 3e-6)
    assert q8(op.view, x.data_ptr(), out.data_ptr(), tokens, code, scratch.data_ptr(), scratch.numel()-1, stream) != 0
