"""Torch bridge gates for the same frozen fixtures as the native CUDA test."""
import pytest

from bench.cuda_mxfp4_sq_fixtures import fixture

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def extension():
    from mfq.kernels.cuda._ext import ext
    return ext()


def tensors(bits, n, k, base=120):
    blob, dense = fixture(bits, n, k, base)
    packed = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone().cuda()
    expected = torch.frombuffer(bytearray(dense), dtype=torch.float32).clone().reshape(n, k)
    return packed, expected


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("base", [0, 1, 120, 251])
def test_decode_bit_exact(extension, bits, base):
    blob, expected = tensors(bits, 128, 256, base)
    for fp32 in (True, False):
        actual = extension.mxfp4_sq_dequant_cuda(blob, bits, 128, 256, base, fp32).cpu()
        dtype, int_dtype = (torch.float32, torch.int32) if fp32 else (torch.float16, torch.int16)
        assert torch.equal(actual.view(int_dtype), expected.to(dtype).view(int_dtype))


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_packed_matmul(extension, bits, dtype):
    for n in (1, 7, 33, 128):
        for k in (32, 96, 256):
            blob, dense = tensors(bits, n, k)
            for m in (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 32, 65):
                values = ((torch.arange(m * k) * 17 + 11) % 65 - 32).float().reshape(m, k) / 32
                x = values.to(device="cuda", dtype=dtype)
                actual = extension.mxfp4_sq_matmul_cuda(blob, x, bits, n, k, 120)
                expected = values.double() @ dense.double().T
                assert actual.shape == (m, n) and actual.dtype == dtype
                torch.testing.assert_close(actual.cpu().double(), expected, rtol=.006, atol=.02)


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 16])
def test_packed_backward_input(extension, bits, dtype, m):
    n, k = 33, 96
    blob, dense = tensors(bits, n, k)
    gradient = torch.randn(m, n, device="cuda", dtype=dtype)
    actual = extension.mxfp4_sq_backward_input_cuda(
        blob, gradient, bits, n, k, 120
    )
    expected = gradient.double().cpu() @ dense.double()
    assert actual.shape == (m, k) and actual.dtype == dtype
    torch.testing.assert_close(
        actual.cpu().double(), expected, rtol=0.006, atol=0.02
    )


@pytest.mark.parametrize("bits", [2, 3])
def test_stream_and_graph_replay(extension, bits):
    blob, dense = tensors(bits, 33, 96)
    for m in range(2, 7):
        x = torch.ones((m, 96), device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            invoke = lambda: extension.mxfp4_sq_matmul_cuda(blob, x, bits, 33, 96, 120)
            for _ in range(3):
                invoke()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = invoke()
            graph.replay()
            graph.replay()
            torch.testing.assert_close(output.cpu().float(), dense.sum(1).expand(m, -1), rtol=.006, atol=.02)
            x.fill_(-1)
            graph.replay()
            torch.testing.assert_close(output.cpu().float(), -dense.sum(1).expand(m, -1), rtol=.006, atol=.02)
        stream.synchronize()


@pytest.mark.parametrize("bits", [2, 3])
def test_invalid_tensor_inputs(extension, bits):
    blob, _ = tensors(bits, 7, 96)
    x = torch.ones((2, 96), device="cuda", dtype=torch.float16)
    invoke = lambda p, a: extension.mxfp4_sq_matmul_cuda(p, a, bits, 7, 96, 120)
    unaligned = torch.empty(blob.numel() + 1, device="cuda", dtype=torch.uint8)[1:]
    unaligned.copy_(blob)
    strided = torch.empty(blob.numel() * 2, device="cuda", dtype=torch.uint8)[::2]
    strided.copy_(blob)
    for bad_blob in (blob.cpu(), blob[:-1], blob.view(1, -1), unaligned, strided):
        with pytest.raises(RuntimeError):
            invoke(bad_blob, x)
    for bad_x in (x.cpu(), x.to(torch.int32), x[:, ::2], x[0], torch.ones((96, 2), device="cuda").T):
        with pytest.raises(RuntimeError):
            invoke(blob, bad_x)
