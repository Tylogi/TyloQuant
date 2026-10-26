"""Synthetic Gemma4 and GLM DSA weights based on the architecture probes."""

import numpy as np

CTX = 64


def gemma4_fixture():
    rng = np.random.default_rng(4404)
    hidden, vocab, heads, kv, head_dim = 32, 32, 2, 1, 16
    intermediate, moe_intermediate = 32, 16
    experts, top_k = 3, 2
    layers = ["sliding_attention", "full_attention"]

    def rand(shape, scale=.05):
        return rng.normal(0, scale, shape).astype(np.float32)

    def make_experts(count, out, width):
        dense = rng.normal(0, .035, (count, out, width)).astype(np.float32)
        return dense.astype(np.float16)

    def qweight(out, width):
        value = rng.normal(0, .05, (out, width)).astype(np.float32)
        return value.astype(np.float16)

    config = dict(
        model_type="gemma4_text", vocab_size=vocab, hidden_size=hidden,
        intermediate_size=intermediate, num_hidden_layers=len(layers),
        num_attention_heads=heads, num_key_value_heads=kv,
        num_global_key_value_heads=kv, head_dim=head_dim, global_head_dim=head_dim,
        max_position_embeddings=CTX, sliding_window=1024,
        layer_types=layers, num_experts=experts, num_experts_per_tok=top_k,
        moe_intermediate_size=moe_intermediate, rms_norm_eps=1e-6,
        attention_k_eq_v=False, tie_word_embeddings=False,
        final_logit_softcapping=30.0, eos_token_id=[1, 2],
        sliding_attention=dict(rope_theta=10000.0),
        full_attention=dict(rope_theta=1000000.0, partial_rotary_factor=0.5),
    )
    outer = dict(model_type="gemma4", text_config=config)
    w = {"model.token_embedding.weight": rand((vocab, hidden), .2).astype(np.float16),
         "model.output.weight": rand((vocab, hidden), .1),
         "model.output_norm.weight": np.zeros(hidden, np.float32)}
    for layer, kind in enumerate(layers):
        p = f"model.block.{layer}."
        a = p + "attention."
        m = p + "mlp."
        w[a + "norm.weight"] = np.zeros(hidden, np.float32)
        w[a + "output_norm.weight"] = np.zeros(hidden, np.float32)
        w[a + "query_norm.weight"] = np.zeros(head_dim, np.float32)
        w[a + "key_norm.weight"] = np.zeros(head_dim, np.float32)
        w[a + "query.weight"] = qweight(heads * head_dim, hidden)
        w[a + "key.weight"] = qweight(kv * head_dim, hidden)
        w[a + "value.weight"] = qweight(kv * head_dim, hidden)
        w[a + "output.weight"] = qweight(hidden, heads * head_dim)
        w[p + "output_scale"] = np.array([1.0], np.float32)
        w[m + "dense.input_norm.weight"] = np.zeros(hidden, np.float32)
        w[m + "output_norm.weight"] = np.zeros(hidden, np.float32)
        w[m + "dense.output_norm.weight"] = np.zeros(hidden, np.float32)
        w[m + "experts.input_norm.weight"] = np.zeros(hidden, np.float32)
        w[m + "experts.output_norm.weight"] = np.zeros(hidden, np.float32)
        w[m + "gate.weight"] = qweight(intermediate, hidden)
        w[m + "up.weight"] = qweight(intermediate, hidden)
        w[m + "down.weight"] = qweight(hidden, intermediate)
        w[m + "experts.gate_up.weight"] = make_experts(
            experts, 2 * moe_intermediate, hidden)
        w[m + "experts.down.weight"] = make_experts(
            experts, hidden, moe_intermediate)
        w[m + "router.weight"] = rand((experts, hidden))
        w[m + "router.norm.weight"] = np.zeros(hidden, np.float32)
        w[m + "router.expert_scale"] = np.ones(experts, np.float32)
    return outer, w


def glm_dsa_fixture():
    from mfq.formats.mfe import MfePool, MfeTensor
    from mfq.formats.nint import NintSpec
    from mfq.quantize.nint_quant import quantize

    rng = np.random.default_rng(5505)
    hidden, vocab = 32, 32
    heads, kv = 64, 64
    q_lora, kv_lora = 16, 512
    nope, rope, vdim, qk = 192, 64, 256, 256
    index_dim, index_heads = 128, 32
    moe_inter, shared_inter, experts, top_k = 16, 16, 3, 2
    dense_inter = 32
    layers = ["sparse", "dense"]
    indexers = ["full", "shared"]

    def rand(shape, scale=.05):
        return rng.normal(0, scale, shape).astype(np.float32)

    def mfe(count, out, width):
        dense = rng.normal(0, .05, (count, out, width)).astype(np.float32)
        return dense.astype(np.float16)

    def headwise_fn(count, out, width):
        dense = rng.normal(0, .05, (count, out, width)).astype(np.float32)
        payload = quantize(dense.reshape(count * out, width), NintSpec(4, 24, 6))
        return MfeTensor(dense.shape,
                         (MfePool(np.arange(count, dtype=np.int32), payload),))

    headwise = headwise_fn
    packed_experts = mfe

    def qweight(out, width):
        value = rng.normal(0, .05, (out, width)).astype(np.float32)
        return value.astype(np.float16)

    config = dict(
        model_type="glm_dsa_text", vocab_size=vocab, hidden_size=hidden,
        intermediate_size=dense_inter, num_hidden_layers=len(layers),
        num_attention_heads=heads, num_key_value_heads=kv, head_dim=qk,
        max_position_embeddings=CTX, rms_norm_eps=1e-5,
        layer_types=["glm_dsa"] * len(layers),
        indexer_types=indexers, mlp_layer_types=layers,
        num_experts=experts, num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter, n_shared_experts=1,
        shared_expert_intermediate_size=shared_inter,
        q_lora_rank=q_lora, kv_lora_rank=kv_lora,
        qk_head_dim=qk, qk_nope_head_dim=nope, qk_rope_head_dim=rope,
        v_head_dim=vdim, index_head_dim=index_dim, index_n_heads=index_heads,
        index_topk=2048, n_group=1, topk_group=1, routed_scaling_factor=2.5,
        norm_topk_prob=True, scoring_func="sigmoid", hidden_act="silu",
        topk_method="noaux_tc",
        attention_bias=False, rope_interleave=True, tie_word_embeddings=False,
        eos_token_id=0)
    outer = dict(model_type="glm_dsa", text_config=config)
    w = {"model.token_embedding.weight": rand((vocab, hidden), .2).astype(
             np.float16),
         "model.output.weight": rand((vocab, hidden), .1),
         "model.output_norm.weight": np.zeros(hidden, np.float32)}
    for layer, (indexer, mlp) in enumerate(zip(indexers, layers)):
        p = f"model.block.{layer}."
        a = p + "attention."
        w[a + "norm.weight"] = np.zeros(hidden, np.float32)
        w[p + "mlp.norm.weight"] = np.zeros(hidden, np.float32)
        w[a + "query_a_norm.weight"] = np.zeros(q_lora, np.float32)
        w[a + "key_value_a_norm.weight"] = np.zeros(kv_lora, np.float32)
        w[a + "query_a.weight"] = qweight(q_lora, hidden)
        w[a + "key_value_a.weight"] = qweight(kv_lora + rope, hidden)
        q_b = [qweight(heads * (nope + rope), q_lora)]
        first = [w[a + "query_a.weight"], w[a + "key_value_a.weight"]]
        if indexer == "full":
            first += [qweight(index_dim, hidden), qweight(index_heads, hidden)]
            q_b.append(qweight(index_heads * index_dim, q_lora))
            w[a + "indexer.key_norm.weight"] = np.zeros(index_dim, np.float32)
            w[a + "indexer.key_norm.bias"] = np.zeros(index_dim, np.float32)
        w[a + "query_b.weight"] = q_b[0]
        if len(q_b) > 1:
            w[a + "indexer.query.weight"] = q_b[1]
        w[a + "indexer.key.weight"] = first[2] if len(first) > 2 else None
        if len(first) > 3:
            w[a + "indexer.score.weight"] = first[3]
        w[a + "latent.query_embedding.weight"] = headwise(
            heads, kv_lora, nope)
        w[a + "latent.output_unembedding.weight"] = headwise(
            heads, vdim, kv_lora)
        w[a + "output.weight"] = qweight(hidden, heads * vdim)
        if mlp == "sparse":
            w[p + "mlp.experts.gate_up.weight"] = packed_experts(
                experts, 2 * moe_inter, hidden)
            w[p + "mlp.experts.down.weight"] = packed_experts(
                experts, hidden, moe_inter)
            w[p + "mlp.router.weight"] = rand((experts, hidden))
            w[p + "mlp.router.bias"] = rand((experts,), .02)
            w[p + "mlp.shared_expert.gate.weight"] = rand(
                (shared_inter, hidden))
            w[p + "mlp.shared_expert.up.weight"] = rand((shared_inter, hidden))
            w[p + "mlp.shared_expert.down.weight"] = rand((hidden, shared_inter))
        else:
            w[p + "mlp.gate.weight"] = qweight(dense_inter, hidden)
            w[p + "mlp.up.weight"] = qweight(dense_inter, hidden)
            w[p + "mlp.down.weight"] = qweight(hidden, dense_inter)
    w = {name: value for name, value in w.items() if value is not None}
    return outer, w
