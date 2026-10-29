from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.tq import TQTensor, fixed_codebook, pack_states, unpack_states  # noqa: E402
from mfq.kernels.cuda.tq import ParallelProjections, Projection, SwiGLU, SwiGLUFFN, weighted_reduce  # noqa: E402
from mfq.quantize.cuda.tq_encoder import encode  # noqa: E402
from mfq.quantize.tq import quantize  # noqa: E402


def fixture(rows, width, columns=False, seed=7, groupsize=512):
    rng = np.random.default_rng(seed)
    stored = (width + 255) // 256 * 256
    bits = (rng.permutation(np.arange(rows)) % 8 + 1).astype(np.uint64)
    desc = np.r_[np.uint64(0), np.cumsum(stored * bits[:-1] // 32)]
    desc |= (bits - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(0.015).view(np.uint16)) << np.uint64(40)
    n = (rows * ((stored + groupsize - 1) // groupsize) + 1) // 2
    scales = np.ones(stored, np.float16) if columns else rng.integers(0, 256, n, dtype=np.uint8)
    data = rng.integers(0, 256, int((stored * bits // 8).sum()), dtype=np.uint8)
    return TQTensor(rows, stored, groupsize, columns, desc, scales, data)


def rms(a, b):
    return (
        ((a.float() - b.float()).square().mean() / b.float().square().mean().clamp_min(1e-20))
        .sqrt()
        .item()
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,width,columns",
    [(19, 257, False), (7, 513, False), (1, 37, True), (273, 273, False), (2048, 4096, False), (4096, 2048, True)],
)
def test_gemv(dtype, rows, width, columns):
    torch.backends.cuda.matmul.allow_tf32 = False
    tensor = fixture(rows, width, columns)
    w = torch.tensor(tensor.dequantize(width), device="cuda").float()
    bias = torch.linspace(-0.02, 0.02, rows, device="cuda")
    op = Projection(tensor, input_columns=width, bias=bias)
    x = torch.randn(width + 1, device="cuda", dtype=dtype)[1:]
    output = torch.full((rows + 2,), 42, device="cuda", dtype=dtype)
    expected = w @ x.float() + bias
    for warps, slots in ((0, 0), (1, 4), (2, 2), (4, 1), (8, 1), (16, 1), (32, 1)):
        op.gemv(x, output[1:-1], warps=warps, rows_per_block=slots)
        assert rms(output[1:-1], expected) < (0.001 if dtype == torch.float16 else 0.01)
        assert output[0] == 42 and output[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("groupsize", [16, 32, 256, 512])
def test_row_scale_boundaries(dtype, groupsize):
    """Exercise all row precisions, independent ring tails and short groups."""
    tensor = fixture(24, 273, groupsize=groupsize)
    x = torch.randn(274, device="cuda", dtype=dtype)[1:]
    op = Projection(tensor, input_columns=273)
    expected = torch.tensor(tensor.dequantize(273), device="cuda").float() @ x.float()
    assert rms(op.gemv(x), expected) < (.004 if dtype == torch.bfloat16 else .001)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width,hidden,output", [(257, 19, 37), (4096, 2048, 4096)])
def test_ffn_graph(dtype, width, hidden, output):
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = [
        fixture(hidden, width, False, 11),
        fixture(hidden, width, True, 22),
        fixture(output, hidden, True, 33),
    ]
    g, u, d = [
        Projection(t, input_columns=k) for t, k in zip(tensors, (width, width, hidden), strict=True)
    ]
    gw, uw, dw = [
        torch.tensor(t.dequantize(k), device="cuda").float()
        for t, k in zip(tensors, (width, width, hidden), strict=True)
    ]
    x = torch.randn(width, device="cuda", dtype=dtype) * 0.2
    y = torch.empty(output, device="cuda", dtype=dtype)
    workspace = torch.empty(hidden, device="cuda", dtype=dtype)
    gr, ur = (gw @ x.float()).to(dtype).float(), (uw @ x.float()).to(dtype).float()
    hr = (ur * torch.nn.functional.silu(gr)).to(dtype)
    expected = dw @ hr.float()
    swiglu, ffn = SwiGLU(g, u), SwiGLUFFN(g, u, d)
    swiglu(x, workspace)
    assert rms(workspace, hr) < 0.01
    ffn(x, y, workspace=workspace)
    assert rms(y, expected) < 0.015
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ffn(x, y, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ffn(x, y, workspace=workspace)
    graph.replay()
    assert rms(y, expected) < 0.015


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_compact_column_scale_alignment(dtype):
    """Odd row counts align column scales to eight, rather than sixteen, bytes."""
    tensor = fixture(19, 273, columns=True)
    op = Projection(tensor, input_columns=273)
    image = torch.tensor(np.frombuffer(tensor.to_bytes(), dtype=np.uint8).copy(), device="cuda")
    scale_offset = 64 + tensor.rows * 8
    op.view.descriptors = image.data_ptr() + 64
    op.view.scales = image.data_ptr() + scale_offset
    op.view.words = image.data_ptr() + scale_offset + tensor.scales.nbytes
    x = torch.randn(273, device="cuda", dtype=dtype)
    expected = torch.tensor(tensor.dequantize(273), device="cuda").float() @ x.float()
    assert rms(op.gemv(x), expected) < (.004 if dtype == torch.bfloat16 else .001)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_parallel_projection_rows_and_graph(dtype):
    tensors = [fixture(rows, 273, columns=columns, groupsize=gs, seed=100 + i)
               for i, (rows, gs, columns) in enumerate(((19, 16, False), (257, 32, True), (7, 512, False)))]
    operators = [Projection(t, input_columns=273) for t in tensors]
    group = ParallelProjections(operators)
    x = torch.randn(3, 273, device="cuda", dtype=dtype) * .2
    total = sum(t.rows for t in tensors)
    storage = torch.full((3 * total + 2,), 42, dtype=dtype, device="cuda")
    out = storage[1:-1].reshape(3, total)
    parts = group(x, out)
    expected = [x.float() @ torch.tensor(t.dequantize(273), device="cuda").float().T for t in tensors]
    for actual, ref in zip(parts, expected, strict=True):
        assert rms(actual, ref) < (.004 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 3e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        group(x, out)
    graph.replay()
    for actual, ref in zip(out.split([t.rows for t in tensors], dim=-1), expected, strict=True):
        assert rms(actual, ref) < (.004 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 3e-5)
    assert storage[0] == 42 and storage[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [2, 3, 4, 5, 6, 7, 17, 32])
def test_parallel_token_reuse_contiguous_outputs(dtype, tokens):
    tensors = [fixture(n, 273, columns=i == 1, groupsize=16, seed=83 + i)
               for i, n in enumerate((1, 19, 65))]
    group = ParallelProjections([Projection(t, input_columns=273) for t in tensors])
    x = torch.randn(tokens * 273 + 1, device="cuda", dtype=dtype)[1:].reshape(tokens, 273) * .2
    storage = torch.full((tokens * group.output_columns + 2,), 42, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        assert group.lib.mfq_tq_parallel_tokens(group.plan.data_ptr(), len(tensors), group.blocks,
            x.data_ptr(), storage[1:-1].data_ptr(), tokens, 273,
            {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype], stream) == 0

    run()
    expected = [(x.float() @ torch.tensor(t.dequantize(273), device="cuda").float().T).ravel()
                for t in tensors]
    tolerance = .004 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 3e-5
    assert rms(storage[1:-1], torch.cat(expected)) < tolerance
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert rms(storage[1:-1], torch.cat(expected)) < tolerance
    assert storage[0] == 42 and storage[-1] == 42


def q8_reference(tensor, x, groups=1):
    width = x.size(-1)
    values = x.float()
    matrix = torch.tensor(tensor.dequantize(width), device=x.device).float()
    if tensor.columns:
        scales = torch.tensor(tensor.scales[:width].astype(np.float32), device=x.device)
        values = values * scales
        matrix = torch.where(scales != 0, matrix / scales, 0)
    padded = torch.nn.functional.pad(values, (0, -width % 16)).reshape(*values.shape[:-1], -1, 16)
    maximum = padded.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127, 1)
    decoded = (padded / scale).round().clamp(-127, 127) * scale
    decoded = decoded.flatten(-2)[..., :width]
    if groups == 1:
        return decoded @ matrix.T
    return torch.einsum("tgk,grk->tgr", decoded, matrix.reshape(groups, -1, width)).flatten(-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("groups,columns", [(1, False), (3, True)])
def test_q8_projection_reference_guards_and_changed_graph(dtype, tokens, groups, columns):
    width, rows = 273, groups * 19
    tensor = fixture(rows, width, columns, groupsize=16)
    bias = torch.linspace(-.01, .01, rows, device="cuda")
    op = Projection(tensor, input_columns=width, bias=bias)
    x = torch.randn(tokens * groups * width + 1, device="cuda", dtype=dtype)[1:]
    x = x.reshape(tokens, groups, width)
    storage = torch.full((tokens * rows + 2,), 42, device="cuda", dtype=dtype)
    size = groups * tokens * ((width + 511) // 512 * 512) * 3 // 2
    workspace = torch.empty(size, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        assert op.lib.mfq_tq_q8_tokens(op.view, x.data_ptr(), storage[1:-1].data_ptr(),
            tokens, groups, 0 if dtype == torch.float16 else 2, workspace.data_ptr(), size, stream) == 0

    def check():
        ref = q8_reference(tensor, x, groups).reshape(tokens, rows) + bias
        assert rms(storage[1:-1].reshape(tokens, rows), ref) < .001
        assert storage[0] == 42 and storage[-1] == 42

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-.75)
    graph.replay()
    check()
    x.zero_()
    graph.replay()
    check()
    assert op.lib.mfq_tq_q8_tokens(op.view, x.data_ptr(), storage[1:-1].data_ptr(),
        tokens, groups, 0 if dtype == torch.float16 else 2, workspace.data_ptr(), size - 1, stream) == -1


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("joined", [0, 2])
@pytest.mark.parametrize("copies", [1, 11])
@pytest.mark.parametrize("column_scaled", [False, True])
def test_q8_parallel_mixed_scales_and_changed_graph(dtype, tokens, joined, copies, column_scaled):
    width = 273
    tensors = [fixture(rows, width, columns=column_scaled and i == 1, groupsize=16, seed=71+i)
               for i, rows in enumerate((1, 19, 65) * copies)]
    group = ParallelProjections([Projection(t, input_columns=width) for t in tensors])
    x = torch.randn(tokens, width, device="cuda", dtype=dtype) * .2
    storage = torch.full((tokens * group.output_columns + 2,), 42, device="cuda", dtype=dtype)
    slots = len(tensors) + 1 if column_scaled else 1
    size = slots * tokens * ((width + 511) // 512 * 512) * 3 // 2
    workspace = torch.empty(size, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        assert group.lib.mfq_tq_q8_parallel_tokens(ct.addressof(group.host_plan), group.plan.data_ptr(), len(tensors), group.blocks,
            x.data_ptr(), storage[1:-1].data_ptr(), tokens, width, 0 if dtype == torch.float16 else 2,
            workspace.data_ptr(), size, joined, sum(t.rows for t in tensors[:joined]), stream) == 0

    def check():
        parts = [q8_reference(t, x) for t in tensors]
        if joined:
            parts = [torch.cat(parts[:joined], -1), *parts[joined:]]
        expected = torch.cat([part.ravel() for part in parts])
        assert rms(storage[1:-1], expected) < .001
        assert storage[0] == 42 and storage[-1] == 42

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-.75)
    graph.replay()
    check()
    assert group.lib.mfq_tq_q8_parallel_tokens(ct.addressof(group.host_plan), group.plan.data_ptr(), len(tensors), group.blocks,
        x.data_ptr(), storage[1:-1].data_ptr(), tokens, width, 0 if dtype == torch.float16 else 2,
        workspace.data_ptr(), size - 1, joined, sum(t.rows for t in tensors[:joined]), stream) == -1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [2, 3, 4, 5, 6, 7, 17])
@pytest.mark.parametrize("groups,width,rows_per_group,columns", [
    (1, 273, 19, False), (3, 1025, 7, True), (1, 2051, 9, False),
])
def test_grouped_token_reuse_tails_and_changed_graph_input(dtype, tokens, groups, width,
                                                        rows_per_group, columns):
    rows = groups * rows_per_group
    tensor = fixture(rows, width, columns=columns, groupsize=16)
    bias = torch.linspace(-.01, .01, rows, device="cuda")
    op = Projection(tensor, input_columns=width, bias=bias)
    x = torch.randn(tokens * groups * width + 1, device="cuda", dtype=dtype)[1:]
    x = x.reshape(tokens, groups, width)
    storage = torch.full((tokens * rows + 2,), 42, device="cuda", dtype=dtype)
    matrix = torch.tensor(tensor.dequantize(width), device="cuda").reshape(groups, rows_per_group, width)
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        assert op.lib.mfq_tq_grouped_tokens(op.view, x.data_ptr(), storage[1:-1].data_ptr(),
            tokens, groups, {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype], stream) == 0

    def check():
        expected = torch.einsum("tgk,grk->tgr", x.float(), matrix.float()).reshape(tokens, rows) + bias
        tolerance = .004 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 3e-5
        assert rms(storage[1:-1].reshape(tokens, rows), expected) < tolerance
        assert storage[0] == 42 and storage[-1] == 42

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-.75)
    graph.replay()
    check()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("groups,columns", [(1, False), (3, True)])
def test_float_prefill_exact_source_precision_and_bounded_workspace(dtype, groups, columns):
    rows, width, tokens = groups * 19, 273, 257
    tensor = fixture(rows, width, columns=columns, groupsize=16, seed=94)
    bias = torch.linspace(-.01, .01, rows, device="cuda")
    op = Projection(tensor, input_columns=width, bias=bias)
    x = torch.randn(tokens, groups, width, device="cuda", dtype=dtype) * .2
    weights = torch.tensor(tensor.dequantize(width), device="cuda").float().reshape(groups, 19, width)
    expected = torch.einsum("tgk,grk->tgr", x.float(), weights).reshape(tokens, rows) + bias
    storage = torch.full((tokens * rows + 2,), 42, device="cuda", dtype=dtype)
    workspace = torch.empty(128 << 10, device="cuda", dtype=torch.uint8)
    handle = torch.cuda.current_blas_handle()
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        assert op.lib.mfq_tq_float_gemm(op.view, x.data_ptr(), storage[1:-1].data_ptr(),
            tokens, groups, 1 if dtype == torch.bfloat16 else 2,
            workspace.data_ptr(), workspace.numel(), handle, stream) == 0

    run()
    tolerance = .004 if dtype == torch.bfloat16 else 3e-5
    assert rms(storage[1:-1].reshape(tokens, rows), expected) < tolerance
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert rms(storage[1:-1].reshape(tokens, rows), expected) < tolerance
    assert storage[0] == 42 and storage[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("groups", [1, 3])
def test_prefill_workspace_large_row_tiles_and_group_tails(dtype, groups):
    rows_per_group, width, tokens = 1031, 33, 129
    rows = rows_per_group * groups
    tensor = fixture(rows, width, columns=True, groupsize=16, seed=1841)
    bias = torch.linspace(-.01, .01, rows, device="cuda")
    op = Projection(tensor, input_columns=width, bias=bias)
    x = torch.randn(tokens, groups, width, device="cuda", dtype=dtype) * .2
    weights = torch.tensor(tensor.dequantize(width), device="cuda").float().reshape(groups, rows_per_group, width)
    expected = torch.einsum("tgk,grk->tgr", x.float(), weights).reshape(tokens, rows) + bias
    storage = torch.full((tokens * rows + 2,), 42, device="cuda", dtype=dtype)
    workspace = torch.empty(8 << 20, device="cuda", dtype=torch.uint8)
    handle, stream = torch.cuda.current_blas_handle(), torch.cuda.current_stream().cuda_stream

    def run():
        args = (op.view, x.data_ptr(), storage[1:-1].data_ptr(), tokens, groups)
        resources = (workspace.data_ptr(), workspace.numel(), handle, stream)
        if dtype == torch.float16:
            status = op.lib.mfq_tq_grouped_gemm(*args, *resources)
        else:
            status = op.lib.mfq_tq_float_gemm(*args, 1 if dtype == torch.bfloat16 else 2, *resources)
        assert status == 0

    run()
    tolerance = .004 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 3e-5
    assert rms(storage[1:-1].reshape(tokens, rows), expected) < tolerance
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert rms(storage[1:-1].reshape(tokens, rows), expected) < tolerance
    assert storage[0] == 42 and storage[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_embedding_row_views_replay_mixed_tables(dtype):
    import ctypes as ct
    from mfq.kernels.cuda.tq import ProjectionView

    columns = 273
    tensors = [fixture(1, columns, columns=i % 2 == 1, groupsize=16 << i, seed=1842 + i) for i in range(3)]
    operators = [Projection(t, input_columns=columns) for t in tensors]
    plan = torch.empty(3 * ct.sizeof(ProjectionView), device="cuda", dtype=torch.uint8)
    storage = torch.full((3 * columns + 2,), 42, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    selected = [0, 2, 0]

    def publish():
        views = (ProjectionView * 3)(*(operators[i].view for i in selected))
        plan.copy_(torch.tensor(np.frombuffer(views, dtype=np.uint8).copy(), device="cuda"))

    def run():
        assert operators[0].lib.mfq_tq_embedding_rows(plan.data_ptr(), 3, columns, storage[1:-1].data_ptr(),
            {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype], stream) == 0

    publish()
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    selected = [2, 1, 2]
    publish()
    graph.replay()
    expected = torch.tensor(np.concatenate([tensors[i].dequantize(columns).ravel() for i in selected]), device="cuda").to(dtype)
    torch.testing.assert_close(storage[1:-1], expected, rtol=0, atol=0)
    assert storage[0] == 42 and storage[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_routed_view_tables_reuse_graph_with_new_expert_ids(dtype):
    from mfq.kernels.cuda.tq import ProjectionView

    experts, routes, width, hidden, output = 3, 3, 273, 19, 33
    operators, matrices, tables = {}, {}, {}
    for role, n, k in (("gate", hidden, width), ("up", hidden, width), ("down", output, hidden)):
        tensors = [fixture(n, k, columns=i == 1, seed=131 + i, groupsize=16) for i in range(experts)]
        operators[role] = [Projection(t, input_columns=k) for t in tensors]
        matrices[role] = [torch.tensor(t.dequantize(k), device="cuda").float() for t in tensors]
        data = (ProjectionView * experts)(*(op.view for op in operators[role]))
        tables[role] = torch.tensor(np.frombuffer(data, dtype=np.uint8).copy(), device="cuda")
    lib = operators["gate"][0].lib
    selected = [1, 0, 1]
    ids = torch.tensor(selected, device="cuda", dtype=torch.int64)
    x = torch.randn(width, device="cuda", dtype=dtype) * .2
    h = torch.empty((routes, hidden), device="cuda", dtype=dtype)
    pairs = torch.empty((1, routes, output), device="cuda", dtype=dtype)
    weights = torch.tensor([[.2, .3, .5]], device="cuda", dtype=torch.float32)
    storage = torch.full((output + 2,), 42, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    dt = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype]

    def run():
        assert lib.mfq_tq_routed_swiglu(tables["gate"].data_ptr(), tables["up"].data_ptr(),
            ids.data_ptr(), x.data_ptr(), h.data_ptr(), routes, hidden, dt, .1, stream) == 0
        assert lib.mfq_tq_routed_down(tables["down"].data_ptr(), ids.data_ptr(), h.data_ptr(),
            pairs.data_ptr(), routes, output, hidden, dt, stream) == 0
        weighted_reduce(pairs, weights, storage[1:-1].reshape(1, output))

    def reference():
        values = []
        for e in selected:
            g = (matrices["gate"][e] @ x.float()).to(dtype).float().clamp_max(.1)
            u = (matrices["up"][e] @ x.float()).to(dtype).float().clamp(-.1, .1)
            value = (torch.nn.functional.silu(g) * u).to(dtype).float()
            values.append((matrices["down"][e] @ value).to(dtype).float())
        return (torch.stack(values) * weights[0, :, None]).sum(0).to(dtype)

    run()
    tolerance = .015 if dtype == torch.bfloat16 else .004
    assert rms(storage[1:-1], reference()) < tolerance
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    selected = [2, 1, 2]
    ids.copy_(torch.tensor(selected, device="cuda", dtype=torch.int64))
    graph.replay()
    assert rms(storage[1:-1], reference()) < tolerance
    assert storage[0] == 42 and storage[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tile", [2, 3, 4])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("pair", [False, True])
def test_indexed_token_reuse_and_updated_graph(dtype, tile, columns, pair, monkeypatch):
    import ctypes as ct
    from mfq.kernels.cuda.tq import IndexedProjectionJob, ProjectionView

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    rows, width = 19, 273
    data = [fixture(rows, width, columns, seed=97 + i, groupsize=16) for i in range(4)]
    operators = [Projection(t, input_columns=width,
        bias=torch.linspace(-.02, .02, rows, device="cuda")) for t in data]
    matrices = [torch.tensor(t.dequantize(width), device="cuda").float() for t in data]
    x = torch.randn((5, width), device="cuda", dtype=dtype) * .2
    output = torch.full((2 * tile * rows + 2,), 42, device="cuda", dtype=dtype)
    jobs = torch.empty(2 * ct.sizeof(IndexedProjectionJob), device="cuda", dtype=torch.uint8)
    selected = [[4, 0, 2, 1], [1, 3, 0, 4]]
    counts = [tile, tile - 1]

    def upload():
        values = (IndexedProjectionJob * 2)(*(IndexedProjectionJob(
            operators[2 * j].view, operators[2 * j + 1].view if pair else ProjectionView(),
            (ct.c_int64 * 4)(*(i * width for i in selected[j])),
            (ct.c_int64 * 4)(*(1 + (j * tile + t) * rows for t in range(4))), counts[j])
            for j in range(2)))
        jobs.copy_(torch.from_numpy(np.frombuffer(bytes(values), dtype=np.uint8).copy()).to("cuda"))

    def reference():
        expected = torch.full_like(output, 42)
        for j in range(2):
            for t in range(counts[j]):
                value = (matrices[2 * j] @ x[selected[j][t]].float() + operators[2 * j].bias).to(dtype)
                if pair:
                    up = (matrices[2 * j + 1] @ x[selected[j][t]].float() + operators[2 * j + 1].bias).to(dtype)
                    value = (torch.nn.functional.silu(value.float().clamp_max(.1)) * up.float().clamp(-.1, .1)).to(dtype)
                start = 1 + (j * tile + t) * rows
                expected[start:start + rows] = value
        return expected

    lib = operators[0].lib
    dt = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype]
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        if pair:
            assert lib.mfq_tq_indexed_swiglu(jobs.data_ptr(), x.data_ptr(), output.data_ptr(),
                2, rows, dt, tile, .1, stream) == 0
        else:
            assert lib.mfq_tq_indexed(jobs.data_ptr(), x.data_ptr(), output.data_ptr(),
                2, rows, dt, tile, stream) == 0

    upload()
    run()
    tolerance = .02 if dtype == torch.bfloat16 else .004 if dtype == torch.float16 else 2e-5
    expected = reference()
    active = expected != 42
    assert rms(output[active], expected[active]) < tolerance
    assert torch.equal(output[~active], expected[~active])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    selected = [selected[1], selected[0]]
    x.mul_(.5)
    upload()
    graph.replay()
    expected = reference()
    active = expected != 42
    assert rms(output[active], expected[active]) < tolerance
    assert torch.equal(output[~active], expected[~active])


@pytest.mark.parametrize("width", [1, 273])
@pytest.mark.parametrize("groupsize", [16, 512])
@pytest.mark.parametrize("tile", [5, 6, 8])
@pytest.mark.parametrize("columns", [False, True])
def test_wide_indexed_projection_tails_and_changed_graph(width, groupsize, tile, columns):
    import ctypes as ct
    from mfq.kernels.cuda.tq import IndexedWideProjectionJob

    rows = 19
    data = [fixture(rows, width, columns, seed=191 + j, groupsize=groupsize) for j in range(2)]
    operators = [Projection(t, input_columns=width,
        bias=torch.linspace(-.02, .02, rows, device="cuda")) for t in data]
    matrices = [torch.tensor(t.dequantize(width), device="cuda").float() for t in data]
    backing = torch.randn(10 * width + 1, device="cuda", dtype=torch.float16)
    backing.mul_(.2)
    x = backing[1:].reshape(10, width)
    output = torch.full((2 * tile * rows + 2,), 42, device="cuda", dtype=torch.float16)
    jobs = torch.empty(2 * ct.sizeof(IndexedWideProjectionJob), device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream
    lib = operators[0].lib

    def upload(turn):
        output.fill_(42)
        values = (IndexedWideProjectionJob * 2)()
        expected = torch.full_like(output, 42)
        for j in range(2):
            job = values[j]
            job.weight, job.count = operators[j].view, tile - (turn if j == 0 else 1 - turn)
            for t in range(tile):
                selected = (j + turn * 3 + t * 2) % 10
                job.input_offsets[t], job.output_offsets[t] = selected * width, 1 + (j * tile + t) * rows
                if t < job.count:
                    start = job.output_offsets[t]
                    expected[start:start + rows] = matrices[j] @ x[selected].float() + operators[j].bias
        jobs.copy_(torch.from_numpy(np.frombuffer(bytes(values), dtype=np.uint8).copy()).to("cuda"))
        return expected

    def run():
        assert lib.mfq_tq_indexed_wide(jobs.data_ptr(), x.data_ptr(), output.data_ptr(),
            2, rows, 0, tile, stream) == 0

    expected = upload(0)
    run()
    active = expected != 42
    assert rms(output[active], expected[active]) < .001
    assert torch.equal(output[~active], expected[~active])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-.75)
    expected = upload(1)
    graph.replay()
    active = expected != 42
    assert rms(output[active], expected[active]) < .001
    assert torch.equal(output[~active], expected[~active])


@pytest.mark.parametrize("width", [1, 273])
@pytest.mark.parametrize("groupsizes", [(16, 16), (512, 512), (16, 512), (512, 16)])
@pytest.mark.parametrize("columns", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("limit", [0., .1])
def test_wide_indexed_swiglu_mixed_scales_and_changed_graph(width, groupsizes, columns, limit):
    import ctypes as ct
    from mfq.kernels.cuda.tq import IndexedWidePairJob

    rows, tile = 19, 8
    data = [fixture(rows, width, columns[j], seed=193 + j, groupsize=groupsizes[j]) for j in range(2)]
    operators = [Projection(t, input_columns=width,
        bias=torch.linspace(-.02, .02, rows, device="cuda") * (j + 1)) for j, t in enumerate(data)]
    matrices = [torch.tensor(t.dequantize(width), device="cuda").float() for t in data]
    x = torch.randn(10 * width + 1, device="cuda", dtype=torch.float16)[1:].reshape(10, width)
    x.mul_(.2)
    output = torch.full((tile * rows + 2,), 42, device="cuda", dtype=torch.float16)
    jobs = torch.empty(ct.sizeof(IndexedWidePairJob), device="cuda", dtype=torch.uint8)
    stream, lib = torch.cuda.current_stream().cuda_stream, operators[0].lib

    def upload(turn):
        output.fill_(42)
        job = IndexedWidePairJob()
        job.weight, job.second, job.count = operators[0].view, operators[1].view, tile - 3 * turn
        expected = torch.full_like(output, 42)
        for t in range(tile):
            selected = (turn * 3 + t * 2) % 10
            job.input_offsets[t], job.output_offsets[t] = selected * width, 1 + t * rows
            if t < job.count:
                g, u = [(matrices[j] @ x[selected].float() + operators[j].bias).half().float() for j in range(2)]
                if limit:
                    g, u = g.clamp_max(limit), u.clamp(-limit, limit)
                expected[1 + t * rows:1 + (t + 1) * rows] = u * torch.nn.functional.silu(g)
        jobs.copy_(torch.from_numpy(np.frombuffer(bytes(job), dtype=np.uint8).copy()).to("cuda"))
        return expected

    def run():
        assert lib.mfq_tq_indexed_wide_swiglu(jobs.data_ptr(), x.data_ptr(), output.data_ptr(),
            1, rows, 0, tile, limit, stream) == 0

    expected = upload(0)
    run()
    active = expected != 42
    assert rms(output[active], expected[active]) < .0015
    assert torch.equal(output[~active], expected[~active])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-.75)
    expected = upload(1)
    graph.replay()
    active = expected != 42
    assert rms(output[active], expected[active]) < .0015
    assert torch.equal(output[~active], expected[~active])


@pytest.mark.parametrize("bits", range(1, 9))
def test_encoder_ring(bits):
    torch.manual_seed(4817)
    x = torch.randn(3, 256, device="cuda")
    metric = torch.exp(torch.randn_like(x) * 0.2)
    lut = fixed_codebook()
    value, states, _ = encode(x, metric, torch.tensor(lut, device="cuda"), bits)
    raw = states.cpu().numpy().view(np.uint16)
    np.testing.assert_array_equal(unpack_states(pack_states(raw, bits), bits), raw)
    np.testing.assert_array_equal(value.cpu().numpy(), lut[raw])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,routes,width", [(1, 1, 1), (3, 6, 37), (1, 10, 2560), (3, 33, 273)])
def test_route_weighted_output_and_graph(dtype, tokens, routes, width):
    pairs = torch.randn(tokens, routes, width, device="cuda", dtype=dtype)
    weights = torch.randn(tokens, routes, device="cuda", dtype=torch.float32)
    weights[:, 1::3] = 0
    expected = (pairs.float() * weights.unsqueeze(-1)).sum(dim=1).to(dtype)
    storage = torch.full((tokens * width + 2,), 42, device="cuda", dtype=dtype)
    out = storage[1:-1].reshape(tokens, width)
    weighted_reduce(pairs, weights, out)
    tolerance = .008 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=2e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weighted_reduce(pairs, weights, out)
    graph.replay()
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=2e-6)
    assert storage[0] == 42 and storage[-1] == 42
    with pytest.raises(ValueError, match="routing weights"):
        weighted_reduce(pairs, weights.to(torch.float16), out)


@pytest.mark.parametrize("bits", [1, 3, 8])
def test_encoder_matches_cpu_constrained_objective(bits):
    """Independent NumPy recurrence checks the exact constrained path objective."""
    rng = np.random.default_rng(7201 + bits)
    target = rng.normal(size=256).astype(np.float32)
    weight = np.exp(rng.normal(size=256) * 0.2).astype(np.float32)
    lut = fixed_codebook()
    values, _, boundary = encode(
        torch.tensor(target[None], device="cuda"),
        torch.tensor(weight[None], device="cuda"),
        torch.tensor(lut, device="cuda"),
        bits,
    )
    edges = 1 << (16 - bits)
    states = np.arange(65536)
    costs = np.full(edges, np.inf, dtype=np.float32)
    start = boundary.item()
    costs[start] = 0
    for x, m in zip(target, weight, strict=True):
        delta = (lut.astype(np.float32) - x) * m
        # CUDA uses an FP32 fused multiply-add for delta^2 + predecessor cost.
        trial = (delta.astype(np.float64) ** 2 + costs[states >> bits]).astype(np.float32)
        costs = trial.reshape(1 << bits, edges).min(axis=0)
    decoded = values.cpu().numpy()[0]
    actual = (((decoded.astype(np.float64) - target) * weight) ** 2).sum()
    assert actual == pytest.approx(float(costs[start]), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("width,hidden", [(17, 19), (273, 33), (273, 513), (640, 257)])
@pytest.mark.parametrize("limit", [0., .1])
@pytest.mark.parametrize("implementation", ["wmma", "gemm"])
@pytest.mark.parametrize("capacity,counts", [(37, (37, 31, 1)), (160, (160, 137, 1))])
def test_prefill_batched_moe_graph_and_scatter(width, hidden, limit, implementation, capacity, counts):
    import ctypes as ct
    from mfq.kernels.cuda.tq import PrefillMoeJob, library

    if implementation == "wmma" and torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("FP16 WMMA m16n16k16 requires SM70")
    torch.backends.cuda.matmul.allow_tf32 = False
    routes, tokens = 3, (sum(counts) + 2) // 3 + 3
    slots = torch.randperm(tokens * routes, device="cuda")[:sum(counts)].to(torch.int64)
    x = torch.randn(tokens, width, device="cuda", dtype=torch.float16) * .2
    operations, matrices, biases, jobs = [], [], [], []
    first = 0
    for expert, count in enumerate(counts):
        ops, dense, bias = [], [], []
        for role, (rows, columns) in enumerate(((hidden, width), (hidden, width), (width, hidden))):
            tensor = fixture(rows, columns, columns=(expert + role) % 2 == 1,
                             seed=401 + expert * 7 + role, groupsize=16 if role != 1 else 512)
            b = torch.linspace(-.01, .01, rows, device="cuda")
            ops.append(Projection(tensor, input_columns=columns, bias=b))
            dense.append(torch.tensor(tensor.dequantize(columns), device="cuda").float())
            bias.append(b)
        operations.append(ops)  # Retain compressed images and biases through replay.
        matrices.append(dense)
        biases.append(bias)
        jobs.append(PrefillMoeJob(*(op.view for op in ops), first, count))
        first += count
    raw = bytes((PrefillMoeJob * len(jobs))(*jobs))
    plan = torch.tensor(list(raw), device="cuda", dtype=torch.uint8)
    per_row = 2 * max(width, hidden) + 4 * max(width, 2 * hidden) if implementation == "gemm" else 4 * max(width, hidden) + 2 * hidden
    scratch = torch.empty(len(jobs) * capacity * per_row + 32,
                          device="cuda", dtype=torch.uint8)
    decoded = torch.empty(32768, device="cuda", dtype=torch.uint8)
    storage = torch.full((tokens * routes * width + 2,), 42., device="cuda", dtype=torch.float16)
    out = storage[1:-1].reshape(tokens * routes, width)
    lib = library()
    blas = torch.cuda.current_blas_handle() if implementation == "gemm" else None

    def run():
        args = (plan.data_ptr(), slots.data_ptr(), x.data_ptr(), out.data_ptr(),
                len(jobs), capacity, width, hidden, routes, limit)
        stream = torch.cuda.current_stream().cuda_stream
        if implementation == "gemm":
            assert lib.mfq_tq_prefill_moe_gemm(*args, decoded.data_ptr(), decoded.numel(),
                scratch.data_ptr(), scratch.numel(), blas, stream) == 0
        else:
            assert lib.mfq_tq_prefill_moe(*args, scratch.data_ptr(), scratch.numel(), stream) == 0

    def expected():
        result = torch.full_like(out, 42.)
        first = 0
        for expert, count in enumerate(counts):
            local = slots[first:first + count]
            values = x[local // routes].float()
            g = (values @ matrices[expert][0].T + biases[expert][0]).half().float()
            u = (values @ matrices[expert][1].T + biases[expert][1]).half().float()
            if limit:
                g, u = g.clamp_max(limit), u.clamp(-limit, limit)
            h = (torch.nn.functional.silu(g) * u).half().float()
            result[local] = (h @ matrices[expert][2].T + biases[expert][2]).half()
            first += count
        return result

    run()
    assert rms(out[slots], expected()[slots]) < .003
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        run()
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run()
    x.mul_(-.75)
    graph.replay()
    reference = expected()
    assert rms(out[slots], reference[slots]) < .003
    untouched = torch.ones(tokens * routes, device="cuda", dtype=torch.bool)
    untouched[slots] = False
    assert torch.equal(out[untouched], reference[untouched])
    assert storage[0] == 42 and storage[-1] == 42
    if implementation == "wmma":
        assert lib.mfq_tq_prefill_moe(plan.data_ptr(), slots.data_ptr(), x.data_ptr(), out.data_ptr(),
            len(jobs), capacity, width, hidden, routes, limit, scratch.data_ptr(), 0,
            torch.cuda.current_stream().cuda_stream) == -1


@pytest.mark.parametrize("scale_factors", [None, (1.0,)])
def test_mixed_row_quantization_roundtrip(scale_factors):
    torch.manual_seed(123)
    w = torch.randn(8, 257, device="cuda") * 0.02
    h = torch.linspace(0.1, 2, 257, device="cuda")
    tensor = quantize(w, np.arange(1, 9), imatrix=h, groupsize=512,
                      scale_factors=scale_factors, column_scaling=False, search="adaptive")
    np.testing.assert_array_equal(tensor.bits, np.arange(1, 9))
    assert tensor.to_bytes() == TQTensor.from_bytes(tensor.to_bytes()).to_bytes()
    op = Projection(tensor, input_columns=257)
    x = torch.randn(257, dtype=torch.float16, device="cuda")
    expected = torch.tensor(tensor.dequantize(257), device="cuda").float() @ x.float()
    assert rms(op.gemv(x), expected) < 0.001


@pytest.mark.parametrize("bits", range(1, 9))
def test_gpu_packing_matches_format(bits):
    from mfq.quantize.cuda.tq_fitter import pack

    rng = np.random.default_rng(73 + bits)
    payload = rng.integers(0, 256, (7, 32 * bits), dtype=np.uint8)
    states = unpack_states(payload, bits)
    gpu = torch.tensor(states.view(np.int16), device="cuda")
    np.testing.assert_array_equal(pack(gpu, bits).cpu().numpy(), payload.reshape(-1))


@pytest.mark.parametrize("bits", range(1, 5))
@pytest.mark.parametrize("groupsize", [128, 512])
def test_adaptive_fitting_precision(bits, groupsize):
    torch.manual_seed(913)
    weight = torch.randn(9, 513, device="cuda") * 0.02
    weight[0] = 0
    importance = torch.linspace(0.2, 2, 513, device="cuda")
    importance[::7] = 0
    exact = quantize(weight, bits, imatrix=importance, groupsize=groupsize,
                     column_scaling=False, search="full")
    fast = quantize(weight, bits, imatrix=importance, groupsize=groupsize,
                    column_scaling=False, search="adaptive")
    assert fast.to_bytes() == TQTensor.from_bytes(fast.to_bytes()).to_bytes()
    scores = []
    for tensor in (exact, fast):
        decoded = torch.tensor(tensor.dequantize(513), device="cuda").float()
        scores.append(((decoded - weight).square() * importance).sum().item())
    assert scores[1] <= scores[0] * 1.01
    raw = unpack_states(fast.packed.reshape(-1, 32 * bits), bits)
    np.testing.assert_array_equal(raw >> bits, np.roll(raw, 1, axis=1) & ((1 << (16 - bits)) - 1))
