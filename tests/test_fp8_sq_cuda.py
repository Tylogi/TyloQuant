from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.fp8_sq import decode_fp8_sq_codes, parse_fp8_sq_layout
from mfq.quantize.fp8_sq import (
    E4M3_LEGAL_CODES,
    E4M3_VALUES,
    quantize_fp8_128_sq,
    quantize_mxfp8_sq,
)


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def extension():
    from mfq.kernels.cuda._ext import ext

    return ext()


def _fixture(dtype: str):
    rows, columns = 8, 128
    rng = np.random.default_rng(20260912)
    source = rng.choice(E4M3_LEGAL_CODES, size=(rows, columns)).astype(np.uint8)
    q = np.arange(1, 9, dtype=np.uint8)
    if dtype == "MXFP8-SQ":
        tensor = quantize_mxfp8_sq(
            source,
            np.full((rows, columns // 32), 127, dtype=np.uint8),
            block_shape=(1, 32),
            row_q_bits=q,
        )
    else:
        tensor = quantize_fp8_128_sq(
            source,
            np.ones((1, 1), dtype=np.float16),
            row_q_bits=q,
        )
    layout = parse_fp8_sq_layout(dtype, tensor.payload)
    blob = torch.frombuffer(
        bytearray(tensor.payload), dtype=torch.uint8
    ).clone().cuda()
    row_q = torch.tensor(layout.row_q_bits, dtype=torch.uint8, device="cuda")
    row_offsets = torch.tensor(
        layout.row_symbol_byte_offsets[:-1], dtype=torch.int32, device="cuda"
    )
    dense = torch.from_numpy(
        E4M3_VALUES[decode_fp8_sq_codes(tensor)].astype(np.float32)
    )
    return tensor, layout, blob, row_q, row_offsets, dense


def _dequant(extension, dtype: str, fixture, fp32: bool):
    tensor, layout, blob, row_q, row_offsets, _ = fixture
    if dtype == "MXFP8-SQ":
        return extension.mxfp8_sq_dequant_cuda(
            blob,
            row_q,
            row_offsets,
            tensor.output_size,
            tensor.input_size,
            tensor.block_shape[0],
            tensor.block_shape[1],
            tensor.scale_shape[0],
            tensor.scale_shape[1],
            layout.palettes_offset,
            layout.symbols_offset,
            layout.scales_offset,
            fp32,
        )
    return extension.fp8_128_sq_dequant_cuda(
        blob,
        row_q,
        row_offsets,
        tensor.output_size,
        tensor.input_size,
        3,
        layout.palettes_offset,
        layout.symbols_offset,
        layout.scales_offset,
        fp32,
    )


def _matmul(extension, dtype: str, fixture, x):
    tensor, layout, blob, row_q, row_offsets, _ = fixture
    if dtype == "MXFP8-SQ":
        return extension.mxfp8_sq_matmul_cuda(
            blob,
            row_q,
            row_offsets,
            x,
            tensor.output_size,
            tensor.input_size,
            tensor.block_shape[0],
            tensor.block_shape[1],
            tensor.scale_shape[0],
            tensor.scale_shape[1],
            layout.palettes_offset,
            layout.symbols_offset,
            layout.scales_offset,
        )
    return extension.fp8_128_sq_matmul_cuda(
        blob,
        row_q,
        row_offsets,
        x,
        tensor.output_size,
        tensor.input_size,
        3,
        layout.palettes_offset,
        layout.symbols_offset,
        layout.scales_offset,
    )


def _backward_input(extension, dtype: str, fixture, gradient):
    tensor, layout, blob, row_q, row_offsets, _ = fixture
    if dtype == "MXFP8-SQ":
        return extension.mxfp8_sq_backward_input_cuda(
            blob,
            row_q,
            row_offsets,
            gradient,
            tensor.output_size,
            tensor.input_size,
            tensor.block_shape[0],
            tensor.block_shape[1],
            tensor.scale_shape[0],
            tensor.scale_shape[1],
            layout.palettes_offset,
            layout.symbols_offset,
            layout.scales_offset,
        )
    return extension.fp8_128_sq_backward_input_cuda(
        blob,
        row_q,
        row_offsets,
        gradient,
        tensor.output_size,
        tensor.input_size,
        3,
        layout.palettes_offset,
        layout.symbols_offset,
        layout.scales_offset,
    )


@pytest.mark.parametrize("dtype", ("MXFP8-SQ", "FP8-128SQ"))
@pytest.mark.parametrize("fp32", (False, True))
def test_cuda_decode_matches_native_codepoints(extension, dtype: str, fp32: bool):
    fixture = _fixture(dtype)
    actual = _dequant(extension, dtype, fixture, fp32).cpu()
    expected = fixture[-1].to(actual.dtype)
    assert torch.equal(actual.view(torch.int32 if fp32 else torch.int16),
                       expected.view(torch.int32 if fp32 else torch.int16))


@pytest.mark.parametrize("dtype", ("MXFP8-SQ", "FP8-128SQ"))
@pytest.mark.parametrize("activation_dtype", (torch.float16, torch.float32))
def test_cuda_dense_dispatch_covers_decode_and_prefill(
    extension, dtype: str, activation_dtype
):
    fixture = _fixture(dtype)
    dense = fixture[-1].double()
    for rows in (0, 1, 2, 3, 4, 5, 6, 7, 16, 65):
        values = torch.linspace(-1.0, 1.0, max(1, rows * 128), dtype=torch.float32)
        values = values[: rows * 128].reshape(rows, 128)
        actual = _matmul(
            extension,
            dtype,
            fixture,
            values.to(device="cuda", dtype=activation_dtype),
        )
        expected = values.to(activation_dtype).double() @ dense.T
        assert actual.shape == (rows, 8)
        assert actual.dtype == activation_dtype
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=0.006, atol=0.04
        )


@pytest.mark.parametrize("dtype", ("MXFP8-SQ", "FP8-128SQ"))
@pytest.mark.parametrize("activation_dtype", (torch.float16, torch.float32))
def test_cuda_backward_input(extension, dtype: str, activation_dtype):
    fixture = _fixture(dtype)
    gradient = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(3, 8)
    actual = _backward_input(
        extension,
        dtype,
        fixture,
        gradient.to(device="cuda", dtype=activation_dtype),
    )
    expected = gradient.to(activation_dtype).double() @ fixture[-1].double()
    assert actual.shape == (3, 128)
    assert actual.dtype == activation_dtype
    torch.testing.assert_close(
        actual.cpu().double(), expected, rtol=0.006, atol=0.04
    )
