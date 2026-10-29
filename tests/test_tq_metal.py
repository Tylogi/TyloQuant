import numpy as np
import pytest
from pathlib import Path

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from mfq.formats.tq import TQTensor, fixed_codebook
from mfq.kernels.metal.tq import Projection


def fixture(rows, width, columns=False, seed=7, gs=512):
    rng = np.random.default_rng(seed)
    stored = (width + 255) // 256 * 256
    bits = rng.integers(1, 9, rows, dtype=np.uint64)
    desc = np.r_[np.uint64(0), np.cumsum(stored * bits[:-1] // 32)]
    desc |= (bits - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(.015).view(np.uint16)) << np.uint64(40)
    n = (rows * ((stored + gs - 1) // gs) + 1) // 2
    scales = np.ones(stored, np.float16) if columns else rng.integers(0, 256, n, dtype=np.uint8)
    data = rng.integers(0, 256, int((stored * bits // 8).sum()), dtype=np.uint8)
    return TQTensor(rows, stored, gs, columns, desc, scales, data)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows,width,columns,gs", [(19, 257, False, 512), (7, 513, False, 16),
                                                 (1, 37, True, 512), (65, 640, True, 512)])
def test_projection(dtype, rows, width, columns, gs):
    tensor = fixture(rows, width, columns, gs=gs)
    op = Projection(tensor, input_columns=width)
    rng = np.random.default_rng(12)
    x = mx.array(rng.normal(size=(3, width)).astype(np.float32), dtype)
    expected = np.array(x.astype(mx.float32)) @ tensor.dequantize(width).astype(np.float32).T
    actual = np.array(op(x).astype(mx.float32))
    np.testing.assert_allclose(actual, expected, rtol=.02 if dtype == mx.bfloat16 else .002, atol=.006)


@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("gelu,limit", [(False, 0.), (False, 1.), (True, 0.)])
@pytest.mark.parametrize("width", [257, 640])
def test_routed_pair_and_down(columns, gelu, limit, width):
    experts, hidden = 5, 19
    g = fixture(experts * hidden, width, columns, seed=11)
    u = fixture(experts * hidden, width, not columns, seed=22)
    d = fixture(experts * width, hidden, columns, seed=33)
    gate, up, down = [Projection(t, input_columns=k, experts=experts)
                      for t, k in ((g, width), (u, width), (d, hidden))]
    ids = np.array([[0, 1, -1], [1, 4, 0]], np.int32)
    x = mx.array(np.random.default_rng(20).normal(size=(2, width)).astype(np.float16))
    gp, upv = np.array(gate(x, mx.array(ids))), np.array(up(x, mx.array(ids)))
    gv, uv = gp.astype(np.float32), upv.astype(np.float32)
    if limit:
        gv, uv = np.minimum(gv, limit), np.clip(uv, -limit, limit)
    activation = .5 * gv * (1 + np.tanh(np.sqrt(2 / np.pi) * gv * (1 + .044715 * gv * gv))) if gelu else gv / (1 + np.exp(-gv))
    h = (activation * uv).astype(np.float16)
    actual_h = gate.swiglu(up, x, mx.array(ids), limit=limit, gelu=gelu)
    np.testing.assert_allclose(np.array(actual_h), h, rtol=.01, atol=.001)
    result = np.array(down(actual_h, mx.array(ids)))
    weights = d.dequantize(hidden).reshape(experts, width, hidden).astype(np.float32)
    expected = np.zeros(result.shape, np.float32)
    for token in range(2):
        for route in range(3):
            if ids[token, route] >= 0:
                expected[token, route] = weights[ids[token, route]] @ np.array(actual_h)[token, route].astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=.01, atol=.002)


def test_reject_wrong_geometry():
    op = Projection(fixture(8, 37), input_columns=37, experts=2)
    with pytest.raises(ValueError, match="IDs"):
        op(mx.ones((1, 37), mx.float16))
    with pytest.raises(ValueError, match="width"):
        op(mx.ones((1, 38), mx.float16), mx.zeros((1, 1), mx.int32))


def test_fixed_book_all_states():
    kernel = mx.fast.metal_kernel(name="tq_book_all_states", input_names=["states"],
        output_names=["values"], header=(Path(__file__).resolve().parents[1] / "mfq/kernels/metal/tq_decode.h").read_text(),
        source="uint i = thread_position_in_grid.x; ((device half4*)values)[i] = tq_book(((device const uint4*)states)[i]);")
    result = kernel(inputs=[mx.arange(65536, dtype=mx.uint32)], grid=(16384, 1, 1),
        threadgroup=(128, 1, 1), output_shapes=[(65536,)], output_dtypes=[mx.float16])[0]
    np.testing.assert_array_equal(np.array(result), fixed_codebook())


@pytest.mark.parametrize("columns,gs", [(False, 16), (False, 32), (False, 512), (True, 512)])
@pytest.mark.parametrize("bits", range(1, 9))
def test_decode_eight_ring_wrap(columns, gs, bits):
    from bench.tq_decode import random_tensor
    tensor = random_tensor(9, 768, seed=341, bits=(bits, bits), gs=gs, columns=columns)
    kernel = mx.fast.metal_kernel(name="tq_decode_eight_reference", input_names=["words", "desc", "scales", "relative"],
        output_names=["y"], header=(Path(__file__).resolve().parents[1] / "mfq/kernels/metal/tq_decode.h").read_text(),
        source="""
uint row = thread_position_in_grid.y, col = thread_position_in_grid.x * 8u;
TqDecoded8 value = tq_decode8<COLUMNS, 768, GS>(words, desc[row], scales, relative, row, col);
*(device half4*)(y + row * 768u + col) = value.first;
*(device half4*)(y + row * 768u + col + 4u) = value.second;
""")
    from mfq.formats.tq import RELATIVE
    result = kernel(inputs=[mx.array(tensor.packed.view(np.uint32)), mx.array(tensor.descriptors),
                            mx.array(tensor.scales), mx.array(RELATIVE)],
        grid=(96, 9, 1), threadgroup=(32, 1, 1), output_shapes=[(9, 768)], output_dtypes=[mx.float16],
        template=[("COLUMNS", columns), ("GS", gs)])[0]
    np.testing.assert_array_equal(np.array(result), tensor.dequantize())


@pytest.mark.parametrize("mean", [1.1, 2.3, 4.5, 7.9])
def test_fake_tensor_byte_budget(mean):
    from bench.tq_decode import random_tensor
    rows, width = 128, 768
    overhead = 64 + rows * 8 + (rows * 2 + 1) // 2
    target = round(overhead + rows * width * mean / 8)
    tensor = random_tensor(rows, width, seed=42, byte_budget=target)
    assert abs(tensor.nbytes - target) <= width / 16
    assert np.unique(tensor.bits).size > 1
    with pytest.raises(ValueError, match="outside"):
        random_tensor(rows, width, seed=42, byte_budget=overhead)


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("width,columns,gs", [(640, False, 512), (1025, False, 16), (513, True, 512)])
def test_each_precision_ring_wrap(bits, width, columns, gs):
    from bench.tq_decode import random_tensor
    tensor = random_tensor(11, width, seed=132, bits=(bits, bits), gs=gs, columns=columns)
    op = Projection(tensor, input_columns=width)
    x = np.random.default_rng(19).normal(size=(2, width)).astype(np.float32)
    expected = x @ tensor.dequantize(width).astype(np.float32).T
    np.testing.assert_allclose(np.array(op(mx.array(x))), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("width", [2048, 2049])
def test_expansion_projection_tail(dtype, routed, width):
    outputs, experts = 4099, 2
    tensor = fixture(experts * outputs, width, seed=95)
    op = Projection(tensor, input_columns=width, experts=experts)
    ids = np.array([[0, 1], [1, -1]], np.int32)
    shape = (2, 2, width) if routed else (2, width)
    x = mx.array(np.random.default_rng(72).normal(size=shape).astype(np.float32), dtype)
    weights = tensor.dequantize(width).reshape(experts, outputs, width).astype(np.float32)
    values = np.array(x.astype(mx.float32))
    expected = np.zeros((2, 2, outputs), np.float32)
    for token in range(2):
        for route in range(2):
            if ids[token, route] >= 0:
                expected[token, route] = weights[ids[token, route]] @ (values[token, route] if routed else values[token])
    tolerance = .02 if dtype == mx.bfloat16 else .002 if dtype == mx.float16 else 2e-5
    np.testing.assert_allclose(np.array(op(x, mx.array(ids)).astype(mx.float32)), expected,
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_routed_reduce(dtype, columns, shared):
    width, outputs, experts = 513, 27, 5
    op = Projection(fixture(experts * outputs, width, columns, seed=53), input_columns=width, experts=experts)
    ids = mx.array([[0, 1, -1], [1, 4, 5]], mx.int32)
    weights = mx.array([[.2, .3, .5], [.6, .1, .3]])
    shape = (2, width) if shared else (2, 3, width)
    x = mx.array(np.random.default_rng(27).normal(size=shape).astype(np.float32), dtype)
    pairs = np.array(op(x, ids).astype(mx.float32))
    expected = np.zeros((2, outputs), np.float32)
    for route in range(3):
        expected = (expected.astype(np.float64) + pairs[:, route].astype(np.float64)
                    * np.array(weights).astype(np.float64)[:, route, None]).astype(np.float32)
    actual = op.routed_reduce(x, ids, weights)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), np.array(mx.array(expected, dtype).astype(mx.float32)))


def test_routed_reduce_fallback_and_shape():
    op = Projection(fixture(3 * 19, 1281), input_columns=1281, experts=3)
    x = mx.ones((1, 2, 1281), mx.float16)
    ids = mx.array([[0, 2]], mx.int32)
    weights = mx.array([[.3, .7]])
    expected = mx.sum(op(x, ids).astype(mx.float32) * weights[:, :, None], axis=1).astype(x.dtype)
    np.testing.assert_array_equal(np.array(op.routed_reduce(x, ids, weights)), np.array(expected))
    with pytest.raises(ValueError, match="geometry"):
        op.routed_reduce(x, ids, mx.ones((1, 1)))
