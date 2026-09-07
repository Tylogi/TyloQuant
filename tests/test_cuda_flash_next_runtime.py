"""Native Flash-Next state/selection tests, independent of MLX availability.

KDA uses the Metal runtime equations and the original cached-chunk tolerance
(7e-3); CPU equation/state checks additionally use 2e-5 on F32 fixtures.
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import pytest

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def descriptor(x):
    x = np.asarray(x)
    return dict(shape=list(x.shape), data=x.reshape(-1).tolist(),
                dtype="int64" if np.issubdtype(x.dtype,np.integer) else "float32")


@pytest.fixture(scope="module")
def native():
    binary = os.environ.get("MFQ_FLASH_NEXT_NATIVE_TEST")
    if not binary:
        pytest.skip("MFQ_FLASH_NEXT_NATIVE_TEST required")
    proc = subprocess.Popen([binary, "--json"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def call(op, inputs, **params):
        row = dict(op=op, inputs=[descriptor(x) for x in inputs], params=params)
        proc.stdin.write(json.dumps(row, allow_nan=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("native bridge stopped: " + proc.stderr.read())
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return [None if v is None else np.asarray(v["data"], np.float32).reshape(v["shape"])
                for v in result["outputs"]]

    yield call
    proc.stdin.close()
    try:
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
    assert proc.returncode == 0, proc.stderr.read()


def kda_fixture(width, batch=2, scale=.2):
    rng = np.random.default_rng(5301)
    h, heads, kernel, tokens = 16, 2, 4, 7
    channels = heads * width
    def rand(shape, s=.05):
        return rng.normal(scale=s, size=shape).astype(np.float32)
    x = rand((batch,tokens,h), scale)
    w = [rand((channels,h)) for _ in range(3)]
    w += [rand((heads,h)), rand((width,h)), rand((channels,width)), rand((h,channels))]
    conv = np.zeros((3*channels,1,kernel), np.float32)
    conv[:,0,-1] = .8
    conv[:,0,-2] = .1
    w += [conv, rand((width,h)), rand((channels,width)), rand((channels,),.02),
          rand((heads,),.02), np.ones(width, np.float32)]
    return [x,*w], dict(heads=heads, width=width, kernel=kernel)


def kda_reference(inputs, heads, width, kernel):
    x,qw,kw,vw,bw,ga,gb,ow,conv,fa,fb,bias,alog,norm = inputs
    b,t,_ = x.shape
    qkv = np.concatenate((x@qw.T, x@kw.T, x@vw.T), axis=-1)
    joined = np.concatenate((np.zeros((b,kernel-1,heads*width*3),np.float32), qkv),axis=1)
    convolved = np.zeros(qkv.shape,np.float32)
    for tap in range(kernel):
        convolved += joined[:,tap:tap+t] * conv[:,0,tap]
    convolved *= sigmoid(convolved)
    q,k,v = [a.reshape(b,t,heads,width).transpose(0,2,1,3)
             for a in np.split(convolved,3,axis=-1)]
    q = q / np.maximum(np.sqrt(np.sum(q*q,axis=-1,keepdims=True)),1e-6)
    k = k / np.maximum(np.sqrt(np.sum(k*k,axis=-1,keepdims=True)),1e-6)
    forget = -5 * sigmoid(((x@fa.T)@fb.T+bias).reshape(b,t,heads,width)
                          * np.exp(alog).reshape(1,1,heads,1))
    beta = sigmoid(x@bw.T)
    state = np.zeros((b,heads,width,width),np.float32)
    outputs = []
    for i in range(t):
        state *= np.exp(forget[:,i])[...,None]
        predicted = np.sum(state * k[:,:,i,:,None],axis=-2)
        delta = (v[:,:,i]-predicted)*beta[:,i,:,None]
        state += k[:,:,i,:,None]*delta[...,None,:]
        outputs.append(np.sum(state*q[:,:,i,:,None],axis=-2)/math.sqrt(width))
    attended = np.stack(outputs,axis=2)
    normalized = attended / np.sqrt(np.mean(attended**2,axis=-1,keepdims=True)+1e-5)*norm
    gate = ((x@ga.T)@gb.T).reshape(b,t,heads,width).transpose(0,2,1,3)
    y = (normalized*sigmoid(gate)).transpose(0,2,1,3).reshape(b,t,heads*width)@ow.T
    return y, joined[:,-(kernel-1):],state


@pytest.mark.parametrize("width", [4,32,64,128])
def test_kda_equation_and_chunk_state(native,width):
    inputs,params = kda_fixture(width)
    expected = kda_reference(inputs,**params)
    full = native("runtime_kda",inputs,**params,steps=[dict(begin=0,count=7)])
    for a,e in zip(full,expected,strict=True):
        np.testing.assert_allclose(a,e,atol=2e-5,rtol=2e-5)
    chunked = native("runtime_kda",inputs,**params,
                     steps=[dict(begin=0,count=2),dict(begin=2,count=1),dict(begin=3,count=4)])
    np.testing.assert_allclose(np.concatenate(chunked[::3],axis=1),full[0],atol=7e-3,rtol=7e-3)
    for a,e in zip(chunked[-2:],full[-2:],strict=True):
        np.testing.assert_allclose(a,e,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize("width",[4,128])
@pytest.mark.parametrize("action",["rollback","commit"])
def test_kda_speculative_transaction(native,width,action):
    inputs,params = kda_fixture(width,batch=1)
    prefix = [dict(begin=0,count=2)]
    verified = native("runtime_kda",inputs,**params,steps=[*prefix,
        dict(begin=2,count=2,confirmed=1),{action:True},dict(begin=5,count=1)])
    reference = native("runtime_kda",inputs,**params,steps=[*prefix,
        dict(begin=2,count=1 if action=="rollback" else 2),dict(begin=5,count=1)])
    for a,e in zip(verified[-3:],reference[-3:],strict=True):
        np.testing.assert_allclose(a,e,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(verified[3][:,:1],reference[3][:,:1],atol=7e-3,rtol=7e-3)


def test_kda_tiny_l2_norm_and_no_cache(native):
    inputs,params = kda_fixture(4,scale=1e-8)
    expected = kda_reference(inputs,**params)[0]
    result = native("runtime_kda",inputs,**params,steps=[dict(begin=0,count=7,cache=False)])
    np.testing.assert_allclose(result[0],expected,atol=1e-17,rtol=2e-5)
    assert result[1:] == [None,None]


def test_kda_reset_and_invalid_transaction(native):
    inputs,params = kda_fixture(32)
    out = native("runtime_kda",inputs,**params,steps=[dict(begin=0,count=2),
        dict(reset=True),dict(begin=0,count=2)])
    for a,e in zip(out[:3],out[-3:],strict=True):
        np.testing.assert_array_equal(a,e)
    for steps in [[dict(rollback=True)], [dict(begin=0,count=2,confirmed=1,cache=False)],
                  [dict(begin=0,count=2,confirmed=3)],
                  [dict(begin=0,count=2,confirmed=1),dict(begin=2,count=2,confirmed=1)]]:
        with pytest.raises(RuntimeError):
            native("runtime_kda",inputs,**params,steps=steps)


@pytest.mark.parametrize("pool,budget,tail",[(4,8,True),(4,8,False),(2,4,True),(1,3,True)])
@pytest.mark.parametrize("offset,tokens",[(0,11),(5,6)])
def test_pool_selection_visibility_and_tail(native,pool,budget,tail,offset,tokens):
    length = offset+tokens
    rng = np.random.default_rng(5302)
    scores = rng.normal(size=(2,tokens,length//pool)).astype(np.float32)
    actual = native("runtime_select_pooled_blocks",[scores],query_offset=offset,
                    logical_length=length,pool=pool,budget=budget,tail=tail)[0].astype(np.int64)
    assert actual.shape==(2,tokens,budget+(pool-1 if tail else 0))
    for b in range(2):
        for t in range(tokens):
            position = offset+t
            visible = np.flatnonzero(np.arange(length//pool)*pool+pool-1<=position)
            chosen = visible[np.argsort(scores[b,t,visible])[-(budget//pool):]]
            expected = [int(i*pool+j) for i in chosen for j in range(pool)]
            if tail:
                expected += list(range(position+1-(position+1)%pool,position+1))
            got = actual[b,t][actual[b,t]>=0].tolist()
            assert sorted(got)==sorted(expected)
            assert all(i<=position for i in got)
            assert len(got)==len(set(got))


def test_pool_selection_empty_complete_pools_and_errors(native):
    scores=np.empty((1,2,0),np.float32)
    out=native("runtime_select_pooled_blocks",[scores],query_offset=0,logical_length=2,
               pool=4,budget=8,tail=True)[0]
    np.testing.assert_array_equal(out[0,0],[-1]*8+[0,-1,-1])
    np.testing.assert_array_equal(out[0,1],[-1]*8+[0,1,-1])
    with pytest.raises(RuntimeError):
        native("runtime_select_pooled_blocks",[scores],query_offset=0,logical_length=2,
               pool=4,budget=7,tail=True)


def test_sequence_cache_growth_truncate_overwrite_reset(native):
    x=np.arange(2*35*3,dtype=np.float32).reshape(2,35,3)/9
    out=native("runtime_sequence_cache",[x],maximum=35,steps=[dict(begin=0,count=16),
        dict(begin=16,count=18),dict(truncate=17),dict(begin=32,count=3),
        dict(reset=True),dict(begin=4,count=2)])
    for a,e in zip(out,[x[:,:16],x[:,:34],np.concatenate((x[:,:17],x[:,32:35]),axis=1),x[:,4:6]],strict=True):
        np.testing.assert_array_equal(a,e.astype(np.float16).astype(np.float32))
    for steps in [[dict(begin=0,count=35),dict(begin=0,count=1)],
                  [dict(truncate=1)], [dict(begin=0,count=2),dict(truncate=-1)]]:
        with pytest.raises(RuntimeError):
            native("runtime_sequence_cache",[x],maximum=35,steps=steps)
