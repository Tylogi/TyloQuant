"""Metal Flash-Next equations checked independently against NumPy on both CUDA ABIs.

The original seeds and per-operation tolerances from test_metal_flash_next.py
are retained. Native coverage requires MFQ_FLASH_NEXT_NATIVE_TEST to name the
native test executable; the validation controller forbids skipped tests.
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def rounded(x, dtype):
    if x is None or np.issubdtype(np.asarray(x).dtype, np.integer):
        return x
    return torch.from_numpy(np.asarray(x)).to(getattr(torch, dtype)).float().numpy()


def descriptor(x, dtype="float32", noncontiguous=False):
    if x is None:
        return None
    x = np.asarray(x)
    return dict(shape=list(x.shape), data=x.reshape(-1).tolist(),
                dtype="int64" if np.issubdtype(x.dtype, np.integer) else dtype,
                noncontiguous=noncontiguous)


@pytest.fixture(scope="module", params=["native", "torch"])
def run(request):
    if request.param == "native":
        binary = os.environ.get("MFQ_FLASH_NEXT_NATIVE_TEST")
        if not binary:
            pytest.skip("MFQ_FLASH_NEXT_NATIVE_TEST is not configured")
        process = subprocess.Popen([binary, "--json"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def execute(op, *args, dtype="float32", noncontiguous=False,
                    graph=False, replay_inputs=None, **params):
            row = dict(op=op, inputs=[descriptor(x, dtype, noncontiguous) for x in args],
                       params=params, graph=graph)
            if replay_inputs is not None:
                row["replay_inputs"] = [descriptor(x, dtype, noncontiguous) for x in replay_inputs]
            process.stdin.write(json.dumps(row, allow_nan=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Native bridge terminated: " + process.stderr.read())
            result = json.loads(line)
            if "error" in result:
                raise RuntimeError(result["error"])
            values = [None if x is None else np.asarray(x["data"], np.float32).reshape(x["shape"])
                      for x in result["outputs"]]
            return values[0] if len(values) == 1 else values

        yield execute
        process.stdin.close()
        try:
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert process.returncode == 0, process.stderr.read()
    else:
        from mfq.kernels.cuda._ext import ext
        module = ext()
        torch.backends.cuda.matmul.allow_tf32 = False
        stream = torch.cuda.Stream()

        def execute(op, *args, dtype="float32", noncontiguous=False,
                    graph=False, replay_inputs=None, **params):
            def convert(x):
                if x is None:
                    return None
                x = np.asarray(x)
                t = torch.from_numpy(x.copy()).to("cuda", dtype=(torch.int64
                    if np.issubdtype(x.dtype, np.integer) else getattr(torch, dtype)))
                if noncontiguous and t.ndim >= 2:
                    t = t.transpose(-1, -2).contiguous().transpose(-1, -2)
                return t

            with torch.cuda.stream(stream):
                tensors = [convert(x) for x in args]
                fn = getattr(module, op)
                if graph:
                    fn(*tensors, **params)
                    fn(*tensors, **params)
                    stream.synchronize()
                    capture = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(capture, stream=stream):
                        result = fn(*tensors, **params)
                    capture.replay()
                    capture.replay()
                    if replay_inputs is not None:
                        for target, new in zip(tensors, replay_inputs, strict=True):
                            if target is not None:
                                target.copy_(convert(new))
                        capture.replay()
                else:
                    result = fn(*tensors, **params)
                stream.synchronize()
                single = isinstance(result, torch.Tensor)
                tensors_out = [result] if single else result
                values = [None if x is None else x.float().cpu().numpy() for x in tensors_out]
                return values[0] if single else values

        yield execute


def test_grouped_norm_and_gated_residual(run):
    rng = np.random.default_rng(53)
    c, h, rank = 4, 8, 5
    x = rng.normal(size=(1, 3, c*h)).astype(np.float32)
    norm = rng.normal(scale=.1, size=(c*h,)).astype(np.float32)
    down = rng.normal(scale=.1, size=(rank, c*h)).astype(np.float32)
    up = rng.normal(scale=.1, size=(c*h, rank)).astype(np.float32)
    inject = rng.normal(scale=.1, size=(c, c*h)).astype(np.float32)
    grouped = x.reshape(1, 3, c, h)
    normalized = (grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6))
    normalized = normalized.reshape(x.shape) * (1 + norm)
    np.testing.assert_allclose(run("qwen4_grouped_rms_norm", x, norm, group_size=h),
                               normalized, atol=2e-4, rtol=2e-4)
    low = normalized @ down.T / c
    low = low * sigmoid(low)
    mixing = sigmoid(low @ up.T).reshape(1, 3, c, h)
    expected = (mixing * normalized.reshape(1, 3, c, h)).mean(axis=-2)
    gates = 2 * sigmoid(normalized @ inject.T / c)
    actual, residual, actual_gates = run("qwen4_gated_residual_pre", x, norm, down, up,
                                        inject, hidden_size=h, hc_count=c)
    np.testing.assert_array_equal(residual, x)
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(actual_gates, gates, atol=2e-4, rtol=2e-4)
    no_inject = run("qwen4_gated_residual_pre", x, norm, down, up, None,
                    hidden_size=h, hc_count=c)
    np.testing.assert_allclose(no_inject[0], expected, atol=2e-4, rtol=2e-4)
    assert no_inject[2] is None
    branch = rng.normal(size=(1, 3, h)).astype(np.float32)
    post = run("qwen4_gated_residual_post", branch, x, gates, hc_count=c)
    np.testing.assert_allclose(post, x + (branch[..., None, :] * gates[..., :, None]).reshape(x.shape),
                               atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("iterations", [1, 5, 20])
def test_mhc_sinkhorn(run, iterations):
    rng = np.random.default_rng(54)
    c, h = 4, 8
    mix = (2+c)*c
    x = rng.normal(size=(1, 2, c, h)).astype(np.float32)
    function = rng.normal(scale=.08, size=(mix, c*h)).astype(np.float32)
    base = rng.normal(scale=.05, size=mix).astype(np.float32)
    scale = np.asarray([.8, .9, 1.1], np.float32)
    flat = x.reshape(1, 2, -1)
    flat = flat / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-5)
    pre, post, comb = np.split(flat @ function.T, [c, 2*c], axis=-1)
    pre = sigmoid(pre * scale[0] + base[:c]) + 1e-6
    post = 2 * sigmoid(post * scale[1] + base[c:2*c])
    comb = softmax(comb.reshape(1, 2, c, c)*scale[2] + base[2*c:].reshape(c, c)) + 1e-6
    comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
    for _ in range(iterations - 1):
        comb /= comb.sum(axis=-1, keepdims=True) + 1e-6
        comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
    actual = run("glm5_mhc_pre", x, function, base, scale, sinkhorn_iterations=iterations,
                 noncontiguous=True)
    for got, expected in zip(actual, [post, comb, (pre[..., None]*x).sum(axis=-2)], strict=True):
        np.testing.assert_allclose(got, expected, atol=7e-4, rtol=7e-4)
    branch = rng.normal(size=(1, 2, h)).astype(np.float32)
    got = run("glm5_mhc_post", branch, x, post, comb)
    expected = post[..., None]*branch[..., None, :] + np.matmul(comb.swapaxes(-1, -2), x)
    np.testing.assert_allclose(got, expected, atol=8e-4, rtol=8e-4)


@pytest.mark.parametrize("lower_bound", [-5.0, -2.0])
def test_kda_forget_gate(run, lower_bound):
    rng = np.random.default_rng(55)
    hidden, heads, d = 12, 3, 4
    x = rng.normal(size=(2, 5, hidden)).astype(np.float32)
    fa = rng.normal(scale=.2, size=(d, hidden)).astype(np.float32)
    fb = rng.normal(scale=.2, size=(heads*d, d)).astype(np.float32)
    bias = rng.normal(scale=.1, size=heads*d).astype(np.float32)
    log = rng.normal(scale=.1, size=heads).astype(np.float32)
    gate = ((x @ fa.T) @ fb.T + bias).reshape(2, 5, heads, d)
    expected = lower_bound * sigmoid(np.exp(log)[None, None, :, None] * gate)
    got = run("glm5_kda_forget_gate", x, fa, fb, bias, log, num_heads=heads,
              head_dim=d, lower_bound=lower_bound, noncontiguous=True)
    np.testing.assert_allclose(got, expected, atol=2e-3, rtol=1e-3)
    assert np.all(got >= lower_bound) and np.all(got <= 0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_qsa_and_kpool_scores(run, dtype):
    rng = np.random.default_rng(56)
    q = rounded(rng.normal(size=(2,3,4,8)).astype(np.float32), dtype)
    k = rounded(rng.normal(size=(2,5,8)).astype(np.float32), dtype)
    dots = np.maximum(np.einsum("bthd,bpd->bthp", q, k), 0)
    got = run("qsa_block_scores", q, k, dtype=dtype, noncontiguous=True)
    np.testing.assert_allclose(got, dots.sum(axis=2)/math.sqrt(8), atol=4e-3, rtol=2e-3)
    weights = rounded(rng.normal(size=(2,3,4)).astype(np.float32), dtype)
    got = run("glm5_kpool_scores", q, k, weights, dtype=dtype)
    expected = (dots/math.sqrt(8) * (weights/math.sqrt(4))[..., None]).sum(axis=2)
    np.testing.assert_allclose(got, expected, atol=4e-3, rtol=2e-3)


@pytest.mark.parametrize("tokens", [0, 3, 10])
def test_kpool_complete_groups_only(run, tokens):
    rng = np.random.default_rng(56)
    keys = rng.normal(size=(2,tokens,8)).astype(np.float32)
    gates = rng.normal(size=keys.shape).astype(np.float32)
    ape = rng.normal(scale=.1, size=(4,8)).astype(np.float32)
    got = run("glm5_kpool_states", keys, gates, ape, pool_size=4)
    complete = tokens//4
    if not complete:
        assert got.shape == (2,0,8)
    else:
        expected = (softmax(gates[:,:complete*4].reshape(2,complete,4,8)+ape[None,None], axis=2)
                    * keys[:,:complete*4].reshape(2,complete,4,8)).sum(axis=2)
        np.testing.assert_allclose(got, expected, atol=3e-3, rtol=2e-3)


def attention_reference(q, k, v, *, offset=None, indices=None, scale=None):
    b, heads, tokens, d = q.shape
    k = np.repeat(k, heads//k.shape[1], axis=1)
    v = np.repeat(v, heads//v.shape[1], axis=1)
    if indices is not None:
        k, v = k.astype(np.float16).astype(np.float32), v.astype(np.float16).astype(np.float32)
    out = np.zeros((b,tokens,heads,d), np.float32)
    for batch in range(b):
        for t in range(tokens):
            selected = (np.arange(offset+t+1) if indices is None else
                        indices[batch,t][(indices[batch,t] >= 0) & (indices[batch,t] < k.shape[2])])
            if selected.size == 0:
                continue
            scores = np.einsum("hd,hkd->hk", q[batch,:,t], k[batch][:,selected])
            scores *= 1/math.sqrt(d) if scale is None else scale
            out[batch,t] = np.einsum("hk,hkd->hd", softmax(scores), v[batch][:,selected])
    return out


@pytest.mark.parametrize("mla", [False, True])
def test_dense_and_sparse_attention(run, mla):
    rng = np.random.default_rng(57 if mla else 58)
    heads, kv = (3,1) if mla else (4,2)
    q = rng.normal(scale=.2, size=(1,heads,2,128)).astype(np.float32)
    k = rng.normal(scale=.2, size=(1,kv,5,128)).astype(np.float32)
    v = k if mla else rng.normal(scale=.2, size=k.shape).astype(np.float32)
    indices = np.asarray([[[0,2,3,-1],[1,2,4,-1]]], np.int64)
    scale = 1/math.sqrt(64) if mla else None
    args = (q, k[:,0]) if mla else (q, k, v)
    params = {"scale": scale} if mla else {}
    dense = run("glm5_dense_mla_attention" if mla else "qwen4_dense_gqa_attention",
                *args, query_offset=3, **params)
    sparse = run("glm5_sparse_mla_attention" if mla else "qwen4_sparse_gqa_attention",
                 *args, indices, **params)
    np.testing.assert_allclose(dense, attention_reference(q,k,v,offset=3,scale=scale), atol=4e-3, rtol=4e-3)
    np.testing.assert_allclose(sparse, attention_reference(q,k,v,indices=indices,scale=scale), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("mla,width,heads,kv", [(False,256,24,2), (True,512,64,1),
                                               (False,31,4,2), (True,1025,2,1)])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_sparse_published_widths_invalid_and_duplicate_indices(run, mla, width, heads, kv, dtype):
    rng = np.random.default_rng(5801)
    q = rounded(rng.normal(scale=.1, size=(2,heads,2,width)).astype(np.float32), dtype)
    k = rounded(rng.normal(scale=.1, size=(2,kv,8,width)).astype(np.float32), dtype)
    v = k if mla else rounded(rng.normal(scale=.1, size=k.shape).astype(np.float32), dtype)
    indices = np.asarray([[[0,2,4,6],[3,3,-1,8]], [[-1,-1,9,-2],[7,0,2,4]]], np.int64)
    scale = 1/math.sqrt(256) if mla else None
    args = (q,k[:,0]) if mla else (q,k,v)
    params = {"scale": scale} if mla else {}
    got = run("glm5_sparse_mla_attention" if mla else "qwen4_sparse_gqa_attention",
              *args, indices, dtype=dtype, noncontiguous=True, **params)
    np.testing.assert_allclose(got, attention_reference(q,k,v,indices=indices,scale=scale), atol=6e-3, rtol=6e-3)
    np.testing.assert_array_equal(got[1,0], np.zeros_like(got[1,0]))


@pytest.mark.parametrize("dilation,kernel", [(3,4), (1,1)])
def test_ple_cached_chunks(run, dilation, kernel):
    rng = np.random.default_rng(59)
    x = rng.normal(size=(2,7,11)).astype(np.float32)
    w = rng.normal(scale=.2, size=(11,1,kernel)).astype(np.float32)
    full, state = run("qwen4_ple_dilated_conv_silu", x,w,None,dilation=dilation)
    first, middle = run("qwen4_ple_dilated_conv_silu", x[:,:3],w,None,dilation=dilation)
    second, last = run("qwen4_ple_dilated_conv_silu", x[:,3:],w,middle,dilation=dilation)
    np.testing.assert_allclose(np.concatenate([first,second], axis=1), full, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(state, last)
    length = (kernel-1)*dilation
    combined = np.concatenate([np.zeros((2,length,11), np.float32),x], axis=1)
    acc = np.zeros_like(x)
    for tap in range(kernel):
        acc += combined[:,tap*dilation:tap*dilation+7] * w[:,0,tap]
    np.testing.assert_allclose(full, acc*sigmoid(acc), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("op", ["qsa_block_scores", "qwen4_sparse_gqa_attention", "glm5_kda_forget_gate"])
def test_nondefault_stream_graph_changed_input(run, op):
    rng = np.random.default_rng(5901)
    if op == "qsa_block_scores":
        q = rng.normal(size=(1,2,4,8)).astype(np.float32)
        k = rng.normal(size=(1,5,8)).astype(np.float32)
        original, changed, params = [q,k], [-q,k], {}
    elif op == "qwen4_sparse_gqa_attention":
        q = rng.normal(scale=.2, size=(1,4,2,256)).astype(np.float32)
        k = rng.normal(scale=.2, size=(1,2,5,256)).astype(np.float32)
        v = rng.normal(scale=.2, size=k.shape).astype(np.float32)
        indices = np.asarray([[[0,2,3,-1],[1,2,4,-1]]], np.int64)
        original, changed, params = [q,k,v,indices], [-q,k,-v,indices], {}
    else:
        x = rng.normal(size=(1,2,12)).astype(np.float32)
        fa = rng.normal(scale=.2, size=(4,12)).astype(np.float32)
        fb = rng.normal(scale=.2, size=(12,4)).astype(np.float32)
        bias, log = np.zeros(12,np.float32), np.zeros(3,np.float32)
        original, changed = [x,fa,fb,bias,log], [-x,fa,fb,bias,log]
        params = dict(num_heads=3,head_dim=4)
    expected = run(op,*changed,**params)
    got = run(op,*original,graph=True,replay_inputs=changed,**params)
    np.testing.assert_allclose(got,expected,atol=2e-6,rtol=2e-6)


def test_invalid_shapes_fail(run):
    z = lambda *shape: np.zeros(shape,np.float32)
    with pytest.raises(RuntimeError):
        run("qsa_block_scores",z(1,2,3,4),z(1,2,5))
    with pytest.raises(RuntimeError):
        run("glm5_kpool_states",z(1,2,4),z(1,2,4),z(4,4),pool_size=0)
    with pytest.raises(RuntimeError):
        run("qwen4_ple_dilated_conv_silu",z(1,2,4),z(4,1,3),z(1,1,4),dilation=2)
    with pytest.raises(RuntimeError):
        run("qwen4_sparse_gqa_attention",z(1,3,1,4),z(1,2,5,4),z(1,2,5,4),np.zeros((1,1,2),np.int64))
    with pytest.raises(RuntimeError):
        run("qwen4_dense_gqa_attention",z(1,4,2,8),z(1,2,3,8),z(1,2,3,8),query_offset=2)
