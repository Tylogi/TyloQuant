"""Native Flash-Next state/selection tests, independent of MLX availability.

KDA uses the Metal runtime equations and the original cached-chunk tolerance
(7e-3); CPU equation/state checks additionally use 2e-5 on F32 fixtures.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
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

    def call(op, inputs, dtype="float32", graph=False, **params):
        descriptors=[descriptor(x) for x in inputs]
        for d in descriptors:
            if d["dtype"]!="int64":d["dtype"]=dtype
        row = dict(op=op, inputs=descriptors, params=params, graph=graph)
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
    y = (normalized*sigmoid(gate)).transpose(0,2,1,3).reshape(b,t,heads*width).astype(x.dtype).astype(ow.dtype)@ow.T
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


def glm_model_fixture(attention_types, sparse_ffn):
    from mfq.architectures.flash_next import Glm5NextConfig
    rng=np.random.default_rng(5303)
    h,d,heads,rank,nope,vwidth,index=32,32,2,16,16,16,128
    layers=len(attention_types)
    def rand(shape,scale=.05): return rng.normal(scale=scale,size=shape).astype(np.float32)
    c=dict(model_type="glm5_next_text",vocab_size=32,hidden_size=h,intermediate_size=32,
        num_hidden_layers=layers,max_position_embeddings=32,rms_norm_eps=1e-5,
        layer_types=list(attention_types),mlp_layer_types=["sparse" if sparse_ffn else "dense"]*layers,
        mhc=True,mla_use_nope=True,hc_mult=2,hc_eps=1e-6,hc_sinkhorn_iters=5,
        linear_attn_config=dict(num_heads=heads,head_dim=d,short_conv_kernel_size=4,gate_lower_bound=-5,
            kda_layers=[i for i,t in enumerate(attention_types) if t=="linear_attention"],
            full_attn_layers=[i for i,t in enumerate(attention_types) if t!="linear_attention"]),
        q_lora_rank=rank,kv_lora_rank=rank,num_attention_heads=heads,qk_nope_head_dim=nope,qk_rope_head_dim=0,
        v_head_dim=vwidth,index_n_heads=heads,index_head_dim=index,index_topk=4,index_kpool=2,
        index_kpool_always_select_tail=True,indexer_types=["full"]*layers,n_routed_experts=3,
        num_experts_per_tok=2,moe_intermediate_size=16,n_shared_experts=1,routed_scaling_factor=2.5,
        swiglu_limit=10.0,scoring_func="sigmoid",norm_topk_prob=True,num_nextn_predict_layers=0,
        tie_word_embeddings=False,eos_token_id=[0])
    outer=dict(model_type="glm5_next",text_config=c)
    Glm5NextConfig.from_hf_config(outer)
    w={"model.token_embedding.weight":rand((32,h),.2),"model.output.weight":rand((32,h)),
       "model.output_norm.weight":np.ones(h,np.float32)}
    def dense_ffn(p,inner):
        w[p+".gate.weight"]=rand((inner,h));w[p+".up.weight"]=rand((inner,h));w[p+".down.weight"]=rand((h,inner))
    for i,kind in enumerate(attention_types):
        p=f"model.block.{i}"
        for site in ("attention","mlp"):
            prefix=p+f".{site}.mhc.pre"
            w[prefix+".function"]=rand((8,2*h),.02)
            w[prefix+".base"]=rand((8,),.02)
            w[prefix+".scale"]=np.array([.2,.3,.4],np.float32)
            w[p+f".{site}.norm.weight"]=np.ones(h,np.float32)
        if kind=="linear_attention":
            a=p+".linear_attention"
            for proj in ("query","key","value"):
                w[a+f".{proj}.weight"]=rand((heads*d,h))
                conv=np.zeros((heads*d,1,4),np.float32);conv[:,0,-1]=.8;conv[:,0,-2]=.1
                w[a+f".{proj}_conv.weight"]=conv
            for key,shape in {"forget_a":(d,h),"forget_b":(heads*d,d),"beta":(heads,h),
                              "gate_a":(d,h),"gate_b":(heads*d,d),"output":(h,heads*d)}.items():
                w[a+f".{key}.weight"]=rand(shape)
            w[a+".dt_bias"]=rand((heads*d,),.02);w[a+".a"]=rand((heads,),.02)
            w[a+".output_norm.weight"]=np.ones(d,np.float32)
        else:
            a=p+".attention"
            for key,shape in {"query_a":(rank,h),"key_value_a":(rank,h),"query_b":(heads*nope,rank),
                              "output":(h,heads*vwidth),"indexer.query":(heads*index,rank),
                              "indexer.key":(index,h),"indexer.score":(heads,h),
                              "latent.query_embedding":(heads,rank,nope),
                              "latent.output_unembedding":(heads,vwidth,rank)}.items():
                w[a+f".{key}.weight"]=rand(shape)
            for key,size in [("query_a_norm",rank),("key_value_a_norm",rank),("indexer.key_norm",index)]:
                w[a+f".{key}.weight"]=np.ones(size,np.float32)
            w[a+".indexer.key_norm.bias"]=np.zeros(index,np.float32)
            w[a+".indexer.pool.gate"]=rand((index,h))
            w[a+".indexer.pool.position"]=rand((2,index))
        if sparse_ffn:
            w[p+".mlp.experts.gate_up.weight"]=rand((3,32,h))
            w[p+".mlp.experts.down.weight"]=rand((3,h,16))
            w[p+".mlp.router.weight"]=rand((3,h));w[p+".mlp.router.bias"]=rand((3,),.02)
            dense_ffn(p+".mlp.shared_expert",16)
        else: dense_ffn(p+".mlp",32)
    return outer,w


def glm_model_reference(config,w,mtp=None,return_hidden=False):
    c=config["text_config"]
    def sm(x,axis=-1):
        e=np.exp(x-np.max(x,axis=axis,keepdims=True));return e/e.sum(axis=axis,keepdims=True)
    def norm(x,weight):
        f=x.astype(np.float32)
        return (f/np.sqrt(np.mean(f*f,axis=-1,keepdims=True)+c["rms_norm_eps"])*weight).astype(x.dtype)
    def lin(x,p):
        weight=w[p+".weight"]
        return x.astype(weight.dtype)@weight.T
    def hc_pre(x,p):
        f=x.astype(np.float32).reshape(*x.shape[:2],-1)
        f=f/np.sqrt(np.mean(f*f,axis=-1,keepdims=True)+c["rms_norm_eps"])
        logits=f@w[p+".function"].T;base=w[p+".base"];scale=w[p+".scale"]
        pre=sigmoid(logits[...,:2]*scale[0]+base[:2])+c["hc_eps"]
        post=2*sigmoid(logits[...,2:4]*scale[1]+base[2:4])
        mix=sm(logits[...,4:].reshape(*x.shape[:2],2,2)*scale[2]+base[4:].reshape(2,2))+c["hc_eps"]
        mix=mix/(mix.sum(axis=-2,keepdims=True)+c["hc_eps"])
        for _ in range(1,c["hc_sinkhorn_iters"]):
            mix=mix/(mix.sum(axis=-1,keepdims=True)+c["hc_eps"])
            mix=mix/(mix.sum(axis=-2,keepdims=True)+c["hc_eps"])
        return (pre[...,None]*x).sum(axis=-2).astype(x.dtype),post,mix
    def hc_post(branch,x,post,mix):
        return post.astype(x.dtype)[...,None]*branch[...,None,:]+np.swapaxes(mix,-1,-2)@x
    def dense(x,p):
        g=np.minimum(lin(x,p+".gate"),c["swiglu_limit"])
        u=np.clip(lin(x,p+".up"),-c["swiglu_limit"],c["swiglu_limit"])
        return lin(g*sigmoid(g)*u,p+".down")
    def ffn(x,p,kind):
        if kind=="dense":return dense(x,p)
        source=x.reshape(-1,c["hidden_size"]).astype(np.float16)
        scores=sigmoid(lin(source.astype(np.float32),p+".router"))
        ids=np.argsort(-(scores+w[p+".router.bias"]),axis=-1)[:,:c["num_experts_per_tok"]]
        weight=np.take_along_axis(scores,ids,axis=-1)
        weight=weight/np.maximum(weight.sum(axis=-1,keepdims=True),1e-20)*c["routed_scaling_factor"]
        gu=np.einsum("troi,ti->tro",w[p+".experts.gate_up.weight"][ids],source.astype(np.float32))
        g,u=np.split(gu,2,axis=-1);g=np.minimum(g,c["swiglu_limit"]);u=np.clip(u,-c["swiglu_limit"],c["swiglu_limit"])
        pairs=np.einsum("troi,tri->tro",w[p+".experts.down.weight"][ids],g*sigmoid(g)*u)
        return ((pairs*weight[...,None]).sum(axis=1)+dense(source,p+".shared_expert")).reshape(x.shape)
    def kda(x,p):
        inputs=[x]+[w[p+f".{n}.weight"] for n in ("query","key","value","beta","gate_a","gate_b","output")]
        inputs += [np.concatenate([w[p+f".{n}_conv.weight"] for n in ("query","key","value")],axis=0),
                   w[p+".forget_a.weight"],w[p+".forget_b.weight"],w[p+".dt_bias"],w[p+".a"],w[p+".output_norm.weight"]]
        a=c["linear_attn_config"]
        return kda_reference(inputs,a["num_heads"],a["head_dim"],a["short_conv_kernel_size"])[0]
    def mla(x,p):
        b,t,_=x.shape;heads=c["num_attention_heads"];rank=c["kv_lora_rank"];index=c["index_head_dim"]
        qr=norm(lin(x,p+".query_a"),w[p+".query_a_norm.weight"])
        q=lin(qr,p+".query_b").reshape(b,t,heads,c["qk_nope_head_dim"])
        latent=norm(lin(x,p+".key_value_a"),w[p+".key_value_a_norm.weight"]).astype(np.float16).astype(np.float32)
        absorbed=np.einsum("bthi,hoi->btho",q,w[p+".latent.query_embedding.weight"])
        ik=lin(x,p+".indexer.key").astype(np.float16).astype(np.float32)
        centered=ik-ik.mean(axis=-1,keepdims=True)
        ik=(centered/np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)+1e-6)*w[p+".indexer.key_norm.weight"]+
            w[p+".indexer.key_norm.bias"]).astype(np.float16).astype(np.float32)
        gates=(x@w[p+".indexer.pool.gate"].T).astype(np.float16).astype(np.float32)
        pool=c["index_kpool"];pools=t//pool
        probability=sm(gates[:,:pools*pool].reshape(b,pools,pool,index)+w[p+".indexer.pool.position"],axis=-2)
        # The Metal pool rounds probabilities to its cached keys' dtype first.
        probability=probability.astype(np.float16).astype(np.float32)
        pooled=(ik[:,:pools*pool].reshape(b,pools,pool,index)*probability).astype(np.float16)
        pooled=pooled.sum(axis=-2,dtype=np.float32).astype(np.float16).astype(np.float32)
        iq=lin(qr,p+".indexer.query").reshape(b,t,c["index_n_heads"],index)
        dots=np.maximum(np.einsum("bthi,bpi->bthp",iq,pooled),0)/math.sqrt(index)
        scores=(dots*(lin(x,p+".indexer.score")/math.sqrt(c["index_n_heads"]))[...,None]).sum(axis=-2)
        attended=np.zeros((b,t,heads,rank),np.float32)
        for bi in range(b):
            for token in range(t):
                if t<=c["index_topk"]: selected=list(range(token+1))
                else:
                    visible=np.arange((token+1)//pool)
                    chosen=visible[np.argsort(scores[bi,token,visible])[-(c["index_topk"]//pool):]]
                    selected=[int(p*pool+j) for p in chosen for j in range(pool)]
                    selected+=list(range(token+1-(token+1)%pool,token+1))
                if selected:
                    keys=latent[bi,selected]
                    logits=absorbed[bi,token]@keys.T/math.sqrt(c["qk_nope_head_dim"])
                    attended[bi,token]=sm(logits)@keys
        value=np.einsum("bthi,hoi->btho",attended.astype(np.float16).astype(np.float32),w[p+".latent.output_unembedding.weight"])
        return lin(value.reshape(b,t,-1),p+".output")
    if mtp is not None:
        ids,previous,layer,positions=mtp
        positions=np.arange(ids.shape[1]) if positions is None else positions
        embeds=w["model.token_embedding.weight"][ids]
        embeds=np.where((positions==0)[...,None],np.zeros_like(embeds),embeds)
        x=lin(np.concatenate((norm(embeds,w["predictor.embedding_norm.weight"]),
            norm(previous,w["predictor.hidden_norm.weight"])),axis=-1),"predictor.fusion")
        p=f"predictor.block.{layer}"
        residual=x.astype(np.float32)+mla(norm(x,w[p+".attention.norm.weight"]),p+".attention").astype(np.float32)
        dtype=np.float32 if x.dtype==np.float32 else np.float16
        branch=norm(residual,w[p+".mlp.norm.weight"]).astype(dtype)
        multi=residual+ffn(branch,p+".mlp","sparse").astype(np.float32)
        return norm(multi,w["predictor.output_norm.weight"]).astype(dtype),multi
    hidden=w["model.token_embedding.weight"][np.arange(1,8)[None]].astype(np.float16)
    x=np.broadcast_to(hidden[...,None,:],(*hidden.shape[:2],2,hidden.shape[-1])).copy()
    for i,kind in enumerate(c["layer_types"]):
        p=f"model.block.{i}"
        branch,post,mix=hc_pre(x,p+".attention.mhc.pre")
        branch=norm(branch,w[p+".attention.norm.weight"])
        branch=kda(branch,p+".linear_attention") if kind=="linear_attention" else mla(branch,p+".attention")
        x=hc_post(branch,x,post,mix)
        branch,post,mix=hc_pre(x,p+".mlp.mhc.pre")
        branch=ffn(norm(branch,w[p+".mlp.norm.weight"]),p+".mlp",c["mlp_layer_types"][i])
        x=hc_post(branch,x,post,mix)
    normalized=norm(x.mean(axis=-2),w["model.output_norm.weight"])
    return (normalized,normalized) if return_hidden else lin(normalized,"model.output")


def write_glm_fixture(path,config,weights,predictor=False):
    from mfq.formats.io import save
    from mfq.formats.header import FileHeader
    from mfq.formats.assets import MODEL_CONFIG_ASSET,MODEL_GRAPH_ASSET
    graph=dict(schema_version=1,architecture="glm5_next",
        canonical_naming=dict(namespace="mfq.tensor",version=1,component_roots=["model"]),
        topology=dict(text_layers=config["text_config"]["num_hidden_layers"]),
        graph=dict(kind="causal_lm",backbone="glm5_next"),
        components=[dict(kind="text",tensor_root="model",implementation="glm5_next",policy="decoder")],
        capabilities=["text"])
    if predictor:
        graph["components"].append(dict(kind="predictor",tensor_root="predictor",implementation="next_token_prediction",policy="optional"))
        graph["canonical_naming"]["component_roots"].append("predictor")
        graph["capabilities"].append("mtp")
    save(path,FileHeader(version=2,model_arch="glm5_next"),
         {**weights,MODEL_CONFIG_ASSET:json.dumps(config).encode(),MODEL_GRAPH_ASSET:json.dumps(graph).encode()})


def run_glm_fixture(path):
    bridge=os.environ.get("MFQ_FLASH_NEXT_NATIVE_TEST")
    if not bridge:pytest.skip("MFQ_FLASH_NEXT_NATIVE_TEST required")
    binary=Path(bridge).with_name("mfq-decode")
    return subprocess.run([str(binary),"--mfq",str(path),"--ctx-size","32","--check-flash-next"],
                          text=True,capture_output=True,timeout=90)


@pytest.mark.parametrize("types",[("linear_attention",)*2,("deepseek_sparse_attention",)*2,
                                  ("linear_attention","deepseek_sparse_attention")])
@pytest.mark.parametrize("sparse_ffn",[False,True])
def test_glm_native_complete_graph(tmp_path,types,sparse_ffn):
    config,weights=glm_model_fixture(types,sparse_ffn)
    expected=glm_model_reference(config,weights)
    path=tmp_path/"glm-graph.mfq";write_glm_fixture(path,config,weights)
    process=run_glm_fixture(path)
    assert process.returncode==0,process.stdout+process.stderr
    raw=json.loads(next(line.removeprefix("flash_next_check ") for line in process.stdout.splitlines()
                        if line.startswith("flash_next_check ")))
    assert raw.pop("architecture")=="glm5_next"
    actual={k:np.asarray(v["data"],np.float32).reshape(v["shape"]) for k,v in raw.items()}
    np.testing.assert_allclose(actual["full"],expected,atol=2e-3,rtol=2e-3)
    np.testing.assert_allclose(actual["full"],actual["chunked"],atol=2e-2,rtol=2e-2)
    np.testing.assert_array_equal(actual["full"],actual["reset"])
    for action in ("committed","rejected"):
        np.testing.assert_allclose(actual[action],actual[action+"_reference"],atol=2e-3,rtol=2e-3)


@pytest.mark.parametrize("error",["schedule","rope","mhc","pool","norm","topk"])
def test_glm_native_rejects_inconsistent_config(tmp_path,error):
    config,weights=glm_model_fixture(("linear_attention","deepseek_sparse_attention"),False)
    c=config["text_config"]
    if error=="schedule":c["linear_attn_config"]["kda_layers"]=[1]
    elif error=="rope":c["qk_rope_head_dim"]=4
    elif error=="mhc":c["mhc"]=False
    elif error=="pool":c["index_topk"]=3
    elif error=="norm":c["rms_norm_eps"]=0
    elif error=="topk":c["num_experts_per_tok"]=4
    path=tmp_path/"invalid.mfq";write_glm_fixture(path,config,weights)
    result=run_glm_fixture(path)
    assert result.returncode!=0
    assert "flash_next_check " not in result.stdout


def rotary_reference(x,positions,rotary,maximum,sections,interleaved,base=1e7):
    pairs=rotary//2
    f=x.astype(np.float32)
    pos=np.asarray(positions,dtype=np.int32)
    if pos.ndim==1:pos=pos[None,None]
    elif pos.ndim==2:pos=pos[:,None]
    frequencies=np.power(np.float32(base),-np.arange(0,rotary,2,dtype=np.float32)/rotary)
    cs=[];ss=[]
    for j in range(pairs):
        if interleaved:
            axis=1 if j%3==1 and j<sections[1]*3 else 2 if j%3==2 and j<sections[2]*3 else 0
        else:axis=0 if j<sections[0] else 1 if j<sections[0]+sections[1] else 2
        if axis>=pos.shape[0]:axis=0
        angle=np.clip(pos[axis],0,maximum-1).astype(np.float32)*frequencies[j]
        cs.append(np.cos(angle));ss.append(np.sin(angle))
    cosine=np.stack(cs,axis=-1)[:,None];sine=np.stack(ss,axis=-1)[:,None]
    first=f[...,:pairs];second=f[...,pairs:rotary]
    return np.concatenate((first*cosine-second*sine,second*cosine+first*sine,f[...,rotary:]),axis=-1).astype(x.dtype)


@pytest.mark.parametrize("dtype",["float32","float16"])
@pytest.mark.parametrize("axes",[1,3,6])
@pytest.mark.parametrize("interleaved",[False,True])
def test_qwen_rotary_multi_axis_batch_and_partial(native,dtype,axes,interleaved):
    rng=np.random.default_rng(3802)
    x=rng.normal(size=(2,3,5,10)).astype(dtype)
    positions=np.array([-2,1,7,30,42],np.int64)
    if axes>=3:positions=np.stack((positions,positions+1,positions+3))
    if axes==6:positions=np.stack((positions,positions+4),axis=1)
    p=dict(rotary=8,maximum=32,sections=[2,1,1],interleaved=interleaved)
    expected=rotary_reference(x,positions,**p)
    actual=native("runtime_rotary",[x,positions],dtype=dtype,graph=True,**p)[0]
    np.testing.assert_allclose(actual,expected,atol=2e-3 if dtype=="float16" else 2e-5,
                               rtol=2e-3 if dtype=="float16" else 2e-5)
    np.testing.assert_array_equal(actual[...,8:],x[...,8:])


def qsa_fixture(axes,interleaved):
    rng=np.random.default_rng(3802)
    def rand(shape):return rng.normal(scale=.05,size=shape).astype(np.float32)
    hidden,heads,kv,width,ih,iw=16,4,1,32,2,32
    x=rand((2,7,hidden))*4
    w=[rand((heads*2*width,hidden)),rand((kv*width,hidden)),rand((kv*width,hidden)),
       rand((hidden,heads*width)),rand(((ih+1)*iw,hidden)),rand((width,)),rand((width,)),rand((iw,)),rand((iw,))]
    pos=np.arange(7,dtype=np.int64)
    if axes:pos=np.stack((pos,pos+2,pos+4))
    p=dict(heads=heads,kv_heads=kv,width=width,index_heads=ih,index_width=iw,pool=2,budget=4,
           maximum=32,rotary=8,sections=[2,1,1],interleaved=interleaved)
    return [x,*w,pos],p


def qsa_reference(inputs,p,selected=None):
    x,qw,kw,vw,ow,iqkw,qn,kn,iqn,ikn,pos=inputs
    b,t,_=x.shape;h=p["heads"];kh=p["kv_heads"];d=p["width"];ih=p["index_heads"];iw=p["index_width"]
    def norm(a,w):
        f=a.astype(np.float32);return (f/np.sqrt(np.mean(f*f,axis=-1,keepdims=True)+1e-6)*(1+w)).astype(a.dtype)
    def rope(a,positions):return rotary_reference(a,positions,p["rotary"],p["maximum"],p["sections"],p["interleaved"])
    qp=(x@qw.T).reshape(b,t,h,2*d)
    q=rope(norm(qp[...,:d],qn).transpose(0,2,1,3),pos)
    k=rope(norm((x@kw.T).reshape(b,t,kh,d),kn).transpose(0,2,1,3),pos).astype(np.float16).astype(np.float32)
    v=(x@vw.T).reshape(b,t,kh,d).transpose(0,2,1,3).astype(np.float16).astype(np.float32)
    iqk=x@iqkw.T
    iq=rope(norm(iqk[...,:ih*iw].reshape(b,t,ih,iw),iqn).transpose(0,2,1,3),pos).transpose(0,2,1,3)
    raw=iqk[...,ih*iw:].astype(np.float16)
    pools=t//p["pool"]
    pooled=raw[:,:pools*p["pool"]].reshape(b,pools,p["pool"],iw).astype(np.float32).mean(axis=2).astype(np.float16)
    pooled=rope(norm(pooled,ikn)[:,None],pos[...,np.arange(pools)*p["pool"]])[:,0].astype(np.float32)
    scores=np.maximum(np.einsum("bthi,bpi->bthp",iq,pooled),0).sum(axis=2)/math.sqrt(iw)
    attended=np.zeros((b,t,h,d),np.float32)
    for bi in range(b):
        for token in range(t):
            visible=np.arange((token+1)//p["pool"])
            if p.get("assert_distinct_cutoff",False) and len(visible)>p["budget"]//p["pool"]:
                ordered=np.sort(scores[bi,token,visible]);cut=p["budget"]//p["pool"]
                assert ordered[-cut]-ordered[-cut-1]>1e-4,"full-model fixture has an ambiguous sparse cutoff"
            chosen=visible[np.argsort(scores[bi,token,visible])[-(p["budget"]//p["pool"]):]]
            indices=[int(i*p["pool"]+j) for i in chosen for j in range(p["pool"])]
            indices+=list(range(token+1-(token+1)%p["pool"],token+1))
            if selected is not None:
                actual=selected[bi,token].astype(np.int64)
                assert np.all(actual==selected[bi,token])
                indices=actual[actual>=0].tolist()
                assert len(set(indices))==len(indices) and all(i<=token for i in indices)
                complete=[i for i in indices if i<(token+1)//p["pool"]*p["pool"]]
                blocks=sorted(set(i//p["pool"] for i in complete))
                assert set(complete)=={i*p["pool"]+j for i in blocks for j in range(p["pool"])}
                assert len(blocks)==min(len(visible),p["budget"]//p["pool"])
                remaining=[i for i in visible if i not in blocks]
                # argpartition leaves equal-score selection unspecified. Check
                # the top-k property itself, without choosing a tie policy.
                if remaining:
                    assert min(scores[bi,token,blocks])>=max(scores[bi,token,remaining])-1e-5
                tail=list(range(token+1-(token+1)%p["pool"],token+1))
                assert set(indices)-set(complete)==set(tail)
            for head in range(h):
                logits=q[bi,head,token]@k[bi,head//(h//kh),indices].T/math.sqrt(d)
                probs=np.exp(logits-np.max(logits));probs/=probs.sum()
                attended[bi,token,head]=probs@v[bi,head//(h//kh),indices]
    return (attended*sigmoid(qp[...,d:])).reshape(b,t,h*d)@ow.T


@pytest.mark.parametrize("axes",[False,True])
@pytest.mark.parametrize("interleaved",[False,True])
def test_qsa_equation_chunks_and_cache_rejection(native,axes,interleaved):
    inputs,p=qsa_fixture(axes,interleaved)
    full,_scores,selected,_iq,_pooled=native("runtime_qsa",inputs,**p,steps=[dict(begin=0,count=7)],trace=True)
    np.testing.assert_allclose(full,qsa_reference(inputs,p,selected),atol=5e-4,rtol=5e-4)
    parts=native("runtime_qsa",inputs,**p,steps=[dict(begin=0,count=2),dict(begin=2,count=1),dict(begin=3,count=4)])
    np.testing.assert_allclose(np.concatenate(parts,axis=1),full,atol=1.5e-2,rtol=1.5e-2)
    rejected=native("runtime_qsa",inputs,**p,steps=[dict(begin=0,count=4),dict(truncate=3),dict(begin=5,count=1)])[-1]
    expected=native("runtime_qsa",inputs,**p,steps=[dict(begin=0,count=3),dict(begin=5,count=1)])[-1]
    np.testing.assert_allclose(rejected,expected,atol=2e-5,rtol=2e-5)


def gdn_fixture(width,silu_gate):
    rng=np.random.default_rng(3801)
    rand=lambda shape:rng.normal(scale=.05,size=shape).astype(np.float32)
    h,nk,nv,kernel=16,1,2,4
    x=rand((2,7,h))*4;channels=(2*nk+nv)*width
    conv=np.zeros((channels,1,kernel),np.float32);conv[:,0,-1]=.8;conv[:,0,-2]=.1
    a=[x,rand((channels,h)),rand((nv*width,h)),rand((nv,h)),rand((nv,h)),rand((h,nv*width)),
       conv,rand((nv,)),rand((nv,)),np.ones(width,np.float32)]
    return a,dict(key_heads=nk,value_heads=nv,width=width,kernel=kernel,silu_gate=silu_gate)


def gdn_reference(a,p):
    x,qkv,z,alpha,beta,out,conv,dt,alog,norm=a
    b,t,_=x.shape;nk=p["key_heads"];nv=p["value_heads"];d=p["width"];kernel=p["kernel"]
    raw=x@qkv.T;history=np.pad(raw,((0,0),(kernel-1,0),(0,0)))
    convolved=sum(history[:,j:j+t]*conv[:,0,j] for j in range(kernel))
    convolved=convolved*sigmoid(convolved)
    q,k,v=np.split(convolved,[nk*d,2*nk*d],axis=-1)
    q=q.reshape(b,t,nk,d).transpose(0,2,1,3);k=k.reshape(b,t,nk,d).transpose(0,2,1,3)
    q=q/np.maximum(np.sqrt((q*q).sum(-1,keepdims=True)),1e-6)
    k=k/np.maximum(np.sqrt((k*k).sum(-1,keepdims=True)),1e-6)
    q=np.repeat(q,nv//nk,axis=1);k=np.repeat(k,nv//nk,axis=1)
    v=v.reshape(b,t,nv,d).transpose(0,2,1,3)
    gate=x@alpha.T+dt
    rate=np.exp(-np.exp(alog)*(np.maximum(gate,0)+np.log1p(np.exp(-np.abs(gate)))))
    beta=sigmoid(x@beta.T)
    state=np.zeros((b,nv,d,d),np.float32);attended=[]
    for token in range(t):
        kt=k[:,:,token];decay=rate[:,token,:,None]
        projected=(state*kt[...,None]).sum(-2)
        delta=(v[:,:,token]-decay*projected)*beta[:,token,:,None]
        state=decay[...,None]*state+kt[...,None]*delta[...,None,:]
        attended.append((state*q[:,:,token,:,None]).sum(-2)/math.sqrt(d))
    attended=np.stack(attended,axis=2)
    normalized=attended/np.sqrt(np.mean(attended**2,axis=-1,keepdims=True)+1e-6)*norm
    z=(x@z.T).reshape(b,t,nv,d).transpose(0,2,1,3)
    gate=sigmoid(z)*(z if p["silu_gate"] else 1)
    output=(normalized*gate).transpose(0,2,1,3).reshape(b,t,nv*d)@out.T
    return output,history[:,-(kernel-1):],state


@pytest.mark.parametrize("width",[4,32,64,128])
@pytest.mark.parametrize("silu_gate",[False,True])
def test_qwen_gdn_equation_chunk_and_transactions(native,width,silu_gate):
    a,p=gdn_fixture(width,silu_gate)
    full=native("runtime_gdn",a,**p,steps=[dict(begin=0,count=7)])
    for actual,expected in zip(full,gdn_reference(a,p)):
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    chunks=native("runtime_gdn",a,**p,steps=[dict(begin=0,count=2),dict(begin=2,count=1),dict(begin=3,count=4)])
    np.testing.assert_allclose(np.concatenate(chunks[::3],axis=1),full[0],atol=8e-3,rtol=8e-3)
    for action in ("rollback","commit"):
        steps=[dict(begin=0,count=2),dict(begin=2,count=2,confirmed=1),{action:True},dict(begin=5,count=1)]
        got=native("runtime_gdn",a,**p,steps=steps)[-3:]
        ref=native("runtime_gdn",a,**p,steps=[dict(begin=0,count=2),dict(begin=2,count=1 if action=="rollback" else 2),dict(begin=5,count=1)])[-3:]
        for actual,expected in zip(got,ref):np.testing.assert_allclose(actual,expected,atol=8e-3,rtol=8e-3)


def ngram_reference(ids,p):
    b,t=ids.shape;prefix=p["ngram"]-1;heads=p["heads_per_ngram"]
    history=np.pad(ids,((0,0),(prefix,0)),constant_values=p["eos"])
    result=np.empty((b,t,prefix*heads),np.int64)
    for bi in range(b):
        segment=0
        for token in range(t+prefix):
            for ngram in range(2,p["ngram"]+1):
                mixed=0
                for shift in range(ngram):
                    value=int(history[bi,token-shift]) if token-shift>=segment else p["eos"]
                    mixed^=(value*p["multipliers"][shift])&((1<<64)-1)
                signed=mixed if mixed<(1<<63) else mixed-(1<<64)
                if token>=prefix:
                    for head in range(heads):
                        h=(ngram-2)*heads+head
                        result[bi,token-prefix,h]=signed%p["vocab"][h]+p["offsets"][h]
            if history[bi,token]==p["eos"]:segment=token+1
    return result


def ngram_fixture(heads=2):
    p=dict(ngram=3,heads_per_ngram=heads,eos=15,
           multipliers=[-7046029254386353131,6364136223846793005,-4658895280553007687],
           offsets=list(range(0,16*heads,8)),vocab=[5,7]*heads)
    ids=np.array([[1,2,15,3,4,5,6],[8,15,9,10,15,11,12]],np.int64)
    weights=np.arange(heads*2*8*2,dtype=np.float32).reshape(heads*2,8,2)/64
    return ids,weights,p


@pytest.mark.parametrize("heads",[1,2])
def test_qwen_ngram_shards_signed_hash_eos_and_chunks(native,heads):
    ids,weights,p=ngram_fixture(heads)
    expected=ngram_reference(ids,p)
    full,hashed=native("runtime_ngram",[ids,weights],**p,steps=[dict(begin=0,count=7)])
    np.testing.assert_array_equal(hashed,expected)
    np.testing.assert_array_equal(full,weights.reshape(-1,2)[expected].reshape(2,7,-1))
    chunks=native("runtime_ngram",[ids,weights],**p,steps=[dict(begin=0,count=3),dict(begin=3,count=1),dict(begin=4,count=3)])
    np.testing.assert_array_equal(np.concatenate(chunks[::2],axis=1),full)
    np.testing.assert_array_equal(np.concatenate(chunks[1::2],axis=1),expected)


def ple_fixture():
    ids,weights,p=ngram_fixture()
    rng=np.random.default_rng(3804);rand=lambda shape:rng.normal(scale=.05,size=shape).astype(np.float32)
    p.update(hidden=8,streams=2)
    x=rand((2,7,16))*4
    return [x,ids,weights,rand((16,8)),rand((8,8)),rand((16,)),rand((16,)),rand((16,)),rand((3,16)).T.copy()],p


def ple_reference(a,p):
    x,ids,shards,key,value,kn,qn,cn,conv=a;b,t,_=x.shape;h=p["hidden"];c=p["streams"]
    def norm(x,w):
        f=x.astype(np.float32).reshape(b,t,c,h)
        return ((f/np.sqrt(np.mean(f*f,axis=-1,keepdims=True)+1e-6)).reshape(b,t,c*h)*(1+w)).astype(x.dtype)
    embeddings=shards.reshape(-1,shards.shape[-1])[ngram_reference(ids,p)].reshape(b,t,h)
    k=norm(embeddings@key.T,kn).reshape(b,t,c,h);q=norm(x,qn).reshape(b,t,c,h)
    score=(k*q).sum(-1)/math.sqrt(h)
    root=np.sign(score)*np.sqrt(np.maximum(np.abs(score),1e-6))
    gated=(sigmoid(root)[...,None]*(embeddings@value.T)[:,:,None]).reshape(b,t,c*h)
    taps=(conv[:,0] if conv.ndim==3 else conv).T
    normalized=norm(gated,cn);context=(taps.shape[0]-1)*p["ngram"]
    history=np.pad(normalized,((0,0),(context,0),(0,0)))
    convolved=sum(history[:,j*p["ngram"]:j*p["ngram"]+t]*taps[j] for j in range(taps.shape[0]))
    return (gated+convolved*sigmoid(convolved)).astype(x.dtype),history[:,-context:]


@pytest.mark.parametrize("action",["rollback","commit"])
@pytest.mark.parametrize("packed",[False,True])
def test_qwen_ple_equation_eos_chunks_and_transactions(native,action,packed):
    a,p=ple_fixture()
    if packed:a[-1]=a[-1][:,None,:]
    full=native("runtime_ple",a,**p,steps=[dict(begin=0,count=7)])
    for actual,expected in zip(full,ple_reference(a,p)):
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    chunks=native("runtime_ple",a,**p,steps=[dict(begin=0,count=3),dict(begin=3,count=4)])
    np.testing.assert_allclose(np.concatenate(chunks[::2],axis=1),full[0],atol=1e-3,rtol=1e-3)
    got=native("runtime_ple",a,**p,steps=[dict(begin=0,count=2),dict(begin=2,count=2,confirmed=1),{action:True},dict(begin=5,count=1)])[-2:]
    ref=native("runtime_ple",a,**p,steps=[dict(begin=0,count=2),dict(begin=2,count=1 if action=="rollback" else 2),dict(begin=5,count=1)])[-2:]
    for actual,expected in zip(got,ref):np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-3)


def qwen_model_fixture(interval=2,silu_gate=False,ple=True):
    from mfq.architectures.flash_next import Qwen4ExpConfig
    rng=np.random.default_rng(3805);rand=lambda shape,s=.05:rng.normal(scale=s,size=shape).astype(np.float32)
    h,streams,d,heads,kv,ih=16,2,32,2,1,8
    kinds=["full_attention" if (i+1)%interval==0 else "linear_attention" for i in range(2)]
    ple_layers=[i+1 for i,kind in enumerate(kinds) if kind=="linear_attention"] if ple else []
    c=dict(model_type="qwen4_exp_text",vocab_size=32,hidden_size=h,num_hidden_layers=2,
        max_position_embeddings=32,num_attention_heads=heads,num_key_value_heads=kv,head_dim=d,
        partial_rotary_factor=.25,rope_parameters=dict(rope_theta=1e7,mrope_section=[2,1,1],mrope_interleaved=silu_gate),
        rms_norm_eps=1e-6,layer_types=kinds,full_attention_interval=interval,hc_count=streams,hc_lowrank=4,
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=d,linear_value_head_dim=d,
        linear_conv_kernel_dim=4,num_experts=3,num_experts_per_tok=2,moe_intermediate_size=16,
        shared_expert_intermediate_size=16,norm_topk_prob=silu_gate,output_gate_type="silu" if silu_gate else "sigmoid",
        hidden_act="silu",attention_bias=False,indexer_n_heads=ih,indexer_kv_heads=1,indexer_head_dim=d,
        indexer_budget=4,indexer_compress_ratio=2,ple_layer_ids=ple_layers,ple_embed_dim=h,ple_conv_kernel_size=3,
        ngram_size=3,heads_per_ngram=2,ngram_vocab_size_base=8,split_ngram_parts=4,
        make_ngram_vocab_size_divisible_by=1,seed=3805,mtp_num_hidden_layers=0,tie_word_embeddings=False,eos_token_id=3)
    outer=dict(model_type="qwen4_exp",text_config=c);Qwen4ExpConfig.from_hf_config(outer)
    w={"model.token_embedding.weight":rand((32,h),.2),"model.output.weight":rand((32,h))}
    def gr(p,combine=True):
        w[p+".norm.weight"]=rand((streams*h,),.02);w[p+".down.weight"]=rand((4,streams*h));w[p+".up.weight"]=rand((streams*h,4))
        if combine:w[p.removesuffix(".pre")+".post.inject.weight"]=rand((streams,streams*h))
    gr("model.mhc.pre",False)
    for i,kind in enumerate(kinds):
        p=f"model.block.{i}"
        gr(p+".attention.mhc.pre");gr(p+".mlp.mhc.pre")
        m=p+".mlp"
        w[m+".experts.gate_up.weight"]=rand((3,32,h));w[m+".experts.down.weight"]=rand((3,h,16));w[m+".router.weight"]=rand((3,h))
        for key,shape in {"gate":(16,h),"up":(16,h),"down":(h,16),"router":(1,h)}.items():w[m+f".shared_expert.{key}.weight"]=rand(shape)
        if kind=="linear_attention":
            a=p+".linear_attention"
            for key,shape in {"qkv":(4*d,h),"gate":(2*d,h),"alpha":(2,h),"beta":(2,h),"output":(h,2*d)}.items():w[a+f".{key}.weight"]=rand(shape)
            conv=np.zeros((4*d,1,4),np.float32);conv[:,0,-1]=.8;conv[:,0,-2]=.1
            w[a+".conv.weight"]=conv;w[a+".dt_bias"]=rand((2,));w[a+".a"]=rand((2,));w[a+".norm.weight"]=np.ones(d,np.float32)
        else:
            a=p+".attention"
            for key,shape in {"query":(heads*2*d,h),"key":(kv*d,h),"value":(kv*d,h),"output":(h,heads*d),"indexer.query_key":((ih+1)*d,h)}.items():w[a+f".{key}.weight"]=rand(shape)
            for key in ["query_norm","key_norm","indexer.query_norm","indexer.key_norm"]:w[a+f".{key}.weight"]=rand((d,),.02)
        if i+1 in ple_layers:
            a=p+".position_embedding"
            for shard in range(4):w[a+f".ngram.shard.{shard}.weight"]=rand((8,4),.2)
            w[a+".ngram.layer_multipliers"]=np.array([-7046029254386353131,6364136223846793005,-4658895280553007687],np.int64)
            w[a+".ngram.head_offsets"]=np.array([0,8,16,24],np.int64);w[a+".ngram.head_vocab_sizes"]=np.array([5,7,5,7],np.int64)
            w[a+".key.weight"]=rand((streams*h,h));w[a+".value.weight"]=rand((h,h))
            for key in ["key_norm","query_norm","conv_norm"]:w[a+f".{key}.weight"]=rand((streams*h,),.02)
            w[a+".conv.weight"]=rand((3,streams*h)).T[:,None,:].copy()
    return outer,w


def qwen_model_reference(config,w,positions=None,mtp=None,return_hidden=False):
    c=config["text_config"];h=c["hidden_size"];streams=c["hc_count"]
    ids=np.arange(1,8,dtype=np.int64)[None] if mtp is None else mtp[0];b,t=ids.shape
    x=np.tile(w["model.token_embedding.weight"][ids].astype(np.float16),(1,1,streams))
    def lin(x,p):return x.astype(w[p+".weight"].dtype)@w[p+".weight"].T
    def gr(x,p,combine=True):
        f=x.astype(np.float32).reshape(b,t,streams,h)
        normalized=((f/np.sqrt((f*f).mean(-1,keepdims=True)+1e-6)).reshape(x.shape)*(1+w[p+".norm.weight"])).astype(x.dtype)
        low=lin(normalized,p+".down")/streams;low=low*sigmoid(low)
        mixing=sigmoid(lin(low,p+".up"))
        mixed=(mixing.reshape(b,t,streams,h)*normalized.reshape(b,t,streams,h)).mean(-2)
        inject=2*sigmoid(lin(normalized,p.removesuffix(".pre")+".post.inject")/streams) if combine else None
        return mixed,inject
    if mtp is not None:
        def norm(x,p):
            f=x.astype(np.float32)
            return (f/np.sqrt((f*f).mean(-1,keepdims=True)+c["rms_norm_eps"])*(1+w[p+".weight"])).astype(x.dtype)
        e=lin(norm(w["model.token_embedding.weight"][ids],"predictor.embedding_norm"),"predictor.fusion.embedding")
        hidden=norm(mtp[1],"predictor.hidden_norm").reshape(b,t,streams,h)
        x=(lin(hidden,"predictor.fusion.hidden")+e[:,:,None,:]).reshape(b,t,streams*h)
    layers=list(enumerate(c["layer_types"])) if mtp is None else [(mtp[2],"full_attention")]
    for i,kind in layers:
        p=f"{'model' if mtp is None else 'predictor'}.block.{i}"
        if mtp is None and i+1 in c["ple_layer_ids"]:
            a=p+".position_embedding"
            metadata=dict(ngram=3,heads_per_ngram=2,eos=c["eos_token_id"],hidden=h,streams=streams,
                multipliers=w[a+".ngram.layer_multipliers"].tolist(),offsets=w[a+".ngram.head_offsets"].tolist(),vocab=w[a+".ngram.head_vocab_sizes"].tolist())
            args=[x,ids,np.stack([w[a+f".ngram.shard.{j}.weight"] for j in range(4)])]+[w[a+f".{key}.weight"] for key in ["key","value","key_norm","query_norm","conv_norm","conv"]]
            x=x+ple_reference(args,metadata)[0]
        branch,inject=gr(x,p+".attention.mhc.pre")
        if kind=="linear_attention":
            a=p+".linear_attention"
            args=[branch]+[w[a+f".{key}.weight"] for key in ["qkv","gate","alpha","beta","output","conv"]]+[w[a+".dt_bias"],w[a+".a"],w[a+".norm.weight"]]
            branch=gdn_reference(args,dict(key_heads=1,value_heads=2,width=32,kernel=4,silu_gate=c["output_gate_type"]=="silu"))[0]
        else:
            a=p+".attention"
            args=[branch]+[w[a+f".{key}.weight"] for key in ["query","key","value","output","indexer.query_key","query_norm","key_norm","indexer.query_norm","indexer.key_norm"]]+[np.arange(t) if positions is None else positions]
            branch=qsa_reference(args,dict(heads=2,kv_heads=1,width=32,index_heads=8,index_width=32,pool=2,budget=4,maximum=32,rotary=8,sections=[2,1,1],interleaved=c["rope_parameters"]["mrope_interleaved"],assert_distinct_cutoff=True))
        x=x+(branch[:,:,None]*inject[...,None]).reshape(x.shape)
        branch,inject=gr(x,p+".mlp.mhc.pre");source=branch.reshape(-1,h).astype(np.float16)
        a=p+".mlp";logits=lin(source,a+".router");probs=np.exp(logits-logits.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
        selected=np.argsort(-probs,axis=-1)[:,:2];route=np.take_along_axis(probs,selected,axis=-1)
        if c["norm_topk_prob"]:route/=route.sum(-1,keepdims=True)
        routed=[]
        for row in range(b*t):
            value=np.zeros(h,np.float32)
            for rank,expert in enumerate(selected[row]):
                gu=w[a+".experts.gate_up.weight"][expert]@source[row].astype(np.float32);g,u=np.split(gu,2)
                value+=((g*sigmoid(g)*u)@w[a+".experts.down.weight"][expert].T)*route[row,rank]
            routed.append(value)
        g=lin(source,a+".shared_expert.gate");u=lin(source,a+".shared_expert.up")
        shared=sigmoid(lin(source,a+".shared_expert.router"))*lin(g*sigmoid(g)*u,a+".shared_expert.down")
        branch=(np.stack(routed)+shared).reshape(b,t,h)
        x=x+(branch[:,:,None]*inject[...,None]).reshape(x.shape)
    sample=gr(x,"model.mhc.pre" if mtp is None else "predictor.mhc.pre",False)[0]
    return (sample,x) if mtp is not None or return_hidden else lin(sample,"model.output")


def write_qwen_fixture(path,config,weights,predictor=False):
    from mfq.formats.io import save
    from mfq.formats.header import FileHeader
    from mfq.formats.assets import MODEL_CONFIG_ASSET,MODEL_GRAPH_ASSET
    graph=dict(schema_version=1,architecture="qwen4_exp",canonical_naming=dict(namespace="mfq.tensor",version=1,component_roots=["model"]),
        topology=dict(text_layers=2),graph=dict(kind="causal_lm",backbone="qwen4_exp"),
        components=[dict(kind="text",tensor_root="model",implementation="qwen4_exp",policy="decoder")],capabilities=["text"])
    if predictor:
        graph["components"].append(dict(kind="predictor",tensor_root="predictor",implementation="next_token_prediction",policy="optional"))
        graph["canonical_naming"]["component_roots"].append("predictor")
        graph["capabilities"].append("mtp")
    save(path,FileHeader(version=2,model_arch="qwen4_exp"),{**weights,MODEL_CONFIG_ASSET:json.dumps(config).encode(),MODEL_GRAPH_ASSET:json.dumps(graph).encode()})


@pytest.mark.parametrize("interval,ple",[(1,False),(2,False),(2,True),(3,True)])
@pytest.mark.parametrize("silu_gate",[False,True])
def test_qwen_native_complete_graph(tmp_path,interval,ple,silu_gate):
    c,w=qwen_model_fixture(interval,silu_gate,ple);expected=qwen_model_reference(c,w)
    positions=np.stack((np.arange(7),np.arange(7)+2,np.arange(7)+4))
    axis_expected=qwen_model_reference(c,w,positions)
    path=tmp_path/"qwen-graph.mfq";write_qwen_fixture(path,c,w)
    process=run_glm_fixture(path);assert process.returncode==0,process.stdout+process.stderr
    raw=json.loads(next(line.removeprefix("flash_next_check ") for line in process.stdout.splitlines() if line.startswith("flash_next_check ")))
    assert raw.pop("architecture")=="qwen4_exp"
    got={k:np.asarray(v["data"],np.float32).reshape(v["shape"]) for k,v in raw.items()}
    np.testing.assert_allclose(got["full"],expected,atol=2e-3,rtol=2e-3)
    np.testing.assert_allclose(got["axis_full"],axis_expected,atol=2e-3,rtol=2e-3)
    for key in ("chunked","axis_chunked"):
        np.testing.assert_allclose(got[key],got["axis_full" if key.startswith("axis") else "full"],atol=1.5e-2,rtol=1.5e-2)
    np.testing.assert_array_equal(got["full"],got["reset"])
    np.testing.assert_array_equal(got["full"],got["batch_reset"])
    np.testing.assert_allclose(got["batch"],np.repeat(got["full"],2,axis=0),atol=2e-3,rtol=2e-3)
    np.testing.assert_allclose(got["last"],got["full"][:,-1],atol=2e-5,rtol=2e-5)
    for action in ("committed","rejected"):
        np.testing.assert_allclose(got[action],got[action+"_reference"],atol=2e-3,rtol=2e-3)


@pytest.mark.parametrize("error",["schedule","rotary","ple","linear_width","pool","norm","topk","mtp"])
def test_qwen_native_rejects_inconsistent_config(tmp_path,error):
    config,w=qwen_model_fixture();c=config["text_config"]
    if error=="schedule":c["full_attention_interval"]=1
    elif error=="rotary":c["rope_parameters"]["mrope_section"]=[2,2,2]
    elif error=="ple":c["ple_layer_ids"]=[2]
    elif error=="linear_width":c["linear_value_head_dim"]=64
    elif error=="pool":c["indexer_budget"]=3
    elif error=="norm":c["rms_norm_eps"]=0
    elif error=="topk":c["num_experts_per_tok"]=4
    elif error=="mtp":c["mtp"]={"num_hidden_layers":1}
    path=tmp_path/"invalid-qwen.mfq";write_qwen_fixture(path,config,w)
    result=run_glm_fixture(path);assert result.returncode!=0 and "flash_next_check " not in result.stdout


def mtp_fixture(family,layers=2):
    rng=np.random.default_rng(3811 if family=="qwen" else 5311)
    rand=lambda shape:rng.normal(scale=.05,size=shape).astype(np.float32)
    if family=="qwen":
        config,w=qwen_model_fixture(2,True,True)
        _,head=qwen_model_fixture(1,True,False)
        h=config["text_config"]["hidden_size"];streams=config["text_config"]["hc_count"]
        config["text_config"]["mtp_num_hidden_layers"]=layers
        w["predictor.embedding_norm.weight"]=rand((h,));w["predictor.hidden_norm.weight"]=rand((streams*h,))
        w["predictor.fusion.embedding.weight"]=rand((h,h));w["predictor.fusion.hidden.weight"]=rand((h,h))
        for name,value in head.items():
            if name.startswith("model.mhc.pre."):w[name.replace("model.","predictor.",1)]=value.copy()
    else:
        config,w=glm_model_fixture(("linear_attention","deepseek_sparse_attention"),True)
        _,head=glm_model_fixture(("deepseek_sparse_attention",)*2,True)
        h=config["text_config"]["hidden_size"]
        config["text_config"]["num_nextn_predict_layers"]=layers
        for name in ("embedding_norm","hidden_norm","output_norm"):w[f"predictor.{name}.weight"]=1+rand((h,))
        w["predictor.fusion.weight"]=rand((h,2*h))
    for layer in range(layers):
        prefix=f"model.block.{layer}."
        for name,value in head.items():
            if name.startswith(prefix):w[name.replace("model.","predictor.",1)]=value.copy()
    return config,w


def run_mtp_fixture(path):
    bridge=os.environ.get("MFQ_FLASH_NEXT_NATIVE_TEST")
    if not bridge:pytest.skip("MFQ_FLASH_NEXT_NATIVE_TEST required")
    return subprocess.run([str(Path(bridge).with_name("mfq-decode")),"--mfq",str(path),"--ctx-size","32","--check-flash-next-mtp"],
        text=True,capture_output=True,timeout=90)


@pytest.mark.parametrize("family",["qwen","glm"])
@pytest.mark.parametrize("layers",[1,2])
def test_flash_next_mtp_native_equation_and_server_generation(tmp_path,family,layers):
    config,w=mtp_fixture(family,layers);path=tmp_path/"predictor.mfq"
    (write_qwen_fixture if family=="qwen" else write_glm_fixture)(path,config,w,True)
    process=run_mtp_fixture(path);assert process.returncode==0,process.stdout+process.stderr
    raw=json.loads(next(line.removeprefix("flash_next_mtp_check ") for line in process.stdout.splitlines() if line.startswith("flash_next_mtp_check ")))
    array=lambda row:np.array(row["data"],np.float32).reshape(row["shape"])
    previous=array(raw["previous"]);ids=np.arange(1,8,dtype=np.int64)[None]
    reference=qwen_model_reference if family=="qwen" else glm_model_reference
    for i,row in enumerate(raw["layers"]):
        expected=reference(config,w,mtp=(ids,previous,i,None))
        for name,value in zip(("full","multi"),expected):np.testing.assert_allclose(array(row[name]),value,atol=2e-3,rtol=2e-3)
        np.testing.assert_allclose(array(row["logits"]),expected[0]@w["model.output.weight"].T,atol=2e-3,rtol=2e-3)
        for name in ("reset","batch_reset","independent","uncached"):np.testing.assert_array_equal(array(row[name]),array(row["full"]))
        np.testing.assert_allclose(array(row["chunked"]),array(row["full"]),atol=1.5e-2,rtol=1.5e-2)
        np.testing.assert_allclose(array(row["batch"]),np.repeat(array(row["full"]),2,axis=0),atol=2e-3,rtol=2e-3)
        positions=np.arange(7)+3
        if family=="qwen":
            positions=np.stack((positions,positions+2,positions+4))
            axis=reference(config,w,positions=positions,mtp=(ids,previous,i,None))[0]
        else:axis=reference(config,w,mtp=(ids,previous,i,positions))[0]
        np.testing.assert_allclose(array(row["axis"]),axis,atol=2e-3,rtol=2e-3)
    target=reference(config,w,return_hidden=True)
    np.testing.assert_allclose(array(raw["target_normalized"]),target[0],atol=2e-3,rtol=2e-3)
    np.testing.assert_allclose(array(raw["target_raw"]),target[1],atol=2e-3,rtol=2e-3)
    assert len(raw["layers"])==layers and len(raw["greedy"])==12 and raw["cycles"]>0


@pytest.mark.parametrize("family",["qwen","glm"])
@pytest.mark.parametrize("error",["missing_norm","undeclared","partial_layer","wrong_width"])
def test_flash_next_mtp_rejects_incomplete_head(tmp_path,family,error):
    config,w=mtp_fixture(family)
    if error=="missing_norm":del w["predictor.embedding_norm.weight"]
    elif error=="undeclared":config["text_config"]["mtp_num_hidden_layers" if family=="qwen" else "num_nextn_predict_layers"]=0
    elif error=="partial_layer":del w["predictor.block.1.attention.output.weight"]
    else:w["predictor.hidden_norm.weight"]=np.ones(5,np.float32)
    path=tmp_path/"bad-predictor.mfq"
    (write_qwen_fixture if family=="qwen" else write_glm_fixture)(path,config,w,True)
    result=run_mtp_fixture(path)
    assert result.returncode!=0 and "flash_next_mtp_check " not in result.stdout
