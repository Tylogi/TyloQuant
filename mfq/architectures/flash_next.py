"""Strict normalized contracts for Qwen3.8-Flash-Next and GLM-5.3-Flash.

The public repository names do not match the Transformers architecture IDs:
Qwen3.8-Flash-Next is ``qwen4_exp`` and GLM-5.3-Flash is ``glm5_next``.
Keeping that distinction explicit prevents either checkpoint from silently
falling through an older Qwen3.5 or GLM-5.2 implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive(value: object, name: str) -> int:
    result = int(value or 0)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return tuple(str(item) for item in value)


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return tuple(int(item) for item in value)


def _eos_ids(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return _integers(value, "eos_token_id")


@dataclass(frozen=True)
class VisionTowerConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    depth: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    rms_norm_eps: float
    rope_theta: float
    swiglu_limit: float
    image_size: int | None = None
    projection_intermediate_size: int | None = None
    num_position_embeddings: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> VisionTowerConfig:
        image_size = raw.get("image_size")
        projection = raw.get("projection_intermediate_size")
        positions = raw.get("num_position_embeddings")
        rope_raw = raw.get("rope_parameters")
        rope = rope_raw if isinstance(rope_raw, Mapping) else {}
        hidden = _positive(raw.get("hidden_size"), "vision.hidden_size")
        heads = _positive(raw.get("num_heads"), "vision.num_heads")
        if hidden % heads or (hidden // heads) % 4:
            raise ValueError("vision head dimensions must divide into four axial-RoPE parts")
        return cls(
            model_type=str(raw.get("model_type", "")),
            hidden_size=hidden,
            intermediate_size=_positive(raw.get("intermediate_size"), "vision.intermediate_size"),
            depth=_positive(raw.get("depth"), "vision.depth"),
            num_heads=heads,
            in_channels=_positive(raw.get("in_channels", 3), "vision.in_channels"),
            patch_size=_positive(raw.get("patch_size"), "vision.patch_size"),
            temporal_patch_size=_positive(
                raw.get("temporal_patch_size"), "vision.temporal_patch_size"
            ),
            spatial_merge_size=_positive(
                raw.get("spatial_merge_size"), "vision.spatial_merge_size"
            ),
            out_hidden_size=_positive(raw.get("out_hidden_size"), "vision.out_hidden_size"),
            hidden_act=str(raw.get("hidden_act", "silu")),
            attention_bias=bool(raw.get("attention_bias", False)),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
            swiglu_limit=float(raw.get("swiglu_limit", 10.0)),
            image_size=None if image_size is None else _positive(image_size, "vision.image_size"),
            projection_intermediate_size=(
                None
                if projection is None
                else _positive(projection, "vision.projection_intermediate_size")
            ),
            num_position_embeddings=(
                None
                if positions is None
                else _positive(positions, "vision.num_position_embeddings")
            ),
        )


@dataclass(frozen=True)
class Qwen4ExpConfig:
    family: str
    model_type: str
    text_model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rope_sections: tuple[int, ...]
    mrope_interleaved: bool
    max_position_embeddings: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    full_attention_interval: int
    hc_count: int
    hc_lowrank: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    output_gate_type: str
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    split_ngram_parts: int
    make_ngram_vocab_size_divisible_by: int
    seed: int
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    image_token_id: int | None
    video_token_id: int | None
    vision_start_token_id: int | None
    vision_end_token_id: int | None
    vision: VisionTowerConfig | None

    @classmethod
    def from_hf_config(cls, outer: Mapping[str, object]) -> Qwen4ExpConfig:
        outer_type = str(outer.get("model_type", ""))
        text = _object(outer.get("text_config", outer), "text_config")
        text_type = str(text.get("model_type", outer_type))
        if outer_type not in {"qwen4_exp", "qwen4_exp_text"} or text_type != "qwen4_exp_text":
            raise ValueError(f"expected qwen4_exp/qwen4_exp_text, got {outer_type!r}/{text_type!r}")
        layers = _positive(text.get("num_hidden_layers"), "num_hidden_layers")
        layer_types = _strings(text.get("layer_types"), "layer_types")
        if len(layer_types) != layers or set(layer_types) - {
            "linear_attention",
            "full_attention",
        }:
            raise ValueError("Qwen4-Exp layer_types do not describe every decoder layer")
        interval = _positive(text.get("full_attention_interval"), "full_attention_interval")
        for index, layer_type in enumerate(layer_types):
            expected = "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            if layer_type != expected:
                raise ValueError(
                    f"Qwen4-Exp attention schedule differs at layer {index}: "
                    f"{layer_type} != {expected}"
                )
        hidden = _positive(text.get("hidden_size"), "hidden_size")
        heads = _positive(text.get("num_attention_heads"), "num_attention_heads")
        kv_heads = _positive(text.get("num_key_value_heads"), "num_key_value_heads")
        head_dim = _positive(text.get("head_dim"), "head_dim")
        if heads % kv_heads:
            raise ValueError("Qwen4-Exp attention heads must divide KV heads")
        rope = _object(text.get("rope_parameters", {}), "rope_parameters")
        partial = float(text.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0)))
        rotary_dim = int(round(head_dim * partial))
        sections = _integers(rope.get("mrope_section", ()), "mrope_section")
        if rotary_dim <= 0 or rotary_dim % 2 or (sections and sum(sections) * 2 != rotary_dim):
            raise ValueError("Qwen4-Exp partial/multimodal RoPE dimensions disagree")
        hc_count = _positive(text.get("hc_count"), "hc_count")
        ple_embed = _positive(text.get("ple_embed_dim"), "ple_embed_dim")
        ngram_size = _positive(text.get("ngram_size"), "ngram_size")
        heads_per_ngram = _positive(text.get("heads_per_ngram"), "heads_per_ngram")
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        if hc_count < 2 or ple_embed != hidden or ngram_heads <= 0 or ple_embed % ngram_heads:
            raise ValueError("Qwen4-Exp GR/PLE dimensions disagree")
        budget = _positive(text.get("indexer_budget"), "indexer_budget")
        compress = _positive(text.get("indexer_compress_ratio"), "indexer_compress_ratio")
        if budget % compress:
            raise ValueError("Qwen4-Exp QSA budget must divide its compression ratio")
        indexer_kv_heads = _positive(text.get("indexer_kv_heads"), "indexer_kv_heads")
        if indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA currently requires one index KV head")
        linear_key_dim = _positive(text.get("linear_key_head_dim"), "linear_key_head_dim")
        linear_value_dim = _positive(text.get("linear_value_head_dim"), "linear_value_head_dim")
        if linear_key_dim != linear_value_dim:
            raise ValueError("Qwen4-Exp GDN requires equal key/value head dimensions")
        output_gate_type = str(text.get("output_gate_type") or text.get("hidden_act", "silu"))
        if (
            str(text.get("hidden_act", "silu")) != "silu"
            or bool(text.get("attention_bias", False))
            or output_gate_type not in {"sigmoid", "silu"}
        ):
            raise ValueError("unsupported Qwen4-Exp activation/attention semantics")
        ple_layers = _integers(text.get("ple_layer_ids", ()), "ple_layer_ids")
        if any(
            layer_id < 1 or layer_id > layers or layer_types[layer_id - 1] != "linear_attention"
            for layer_id in ple_layers
        ):
            raise ValueError("Qwen4-Exp PLE must use one-indexed linear-attention layers")
        vision_raw = outer.get("vision_config")
        vision = (
            None
            if vision_raw is None
            else VisionTowerConfig.from_mapping(_object(vision_raw, "vision_config"))
        )
        if vision is not None and vision.out_hidden_size != hidden:
            raise ValueError("Qwen4-Exp vision output width must equal text hidden_size")
        if vision is not None and (
            vision.hidden_act != "gelu_pytorch_tanh"
            or vision.num_position_embeddings is None
            or vision.attention_bias
        ):
            raise ValueError("unsupported Qwen4-Exp vision-tower semantics")
        mtp_count = int(text.get("mtp_num_hidden_layers", 0) or 0)
        mtp_raw = text.get("mtp")
        if isinstance(mtp_raw, Mapping):
            nested_count = int(mtp_raw.get("num_hidden_layers", mtp_count) or 0)
            if nested_count != mtp_count:
                raise ValueError("Qwen4-Exp MTP layer counts disagree")
        return cls(
            family="qwen4_exp",
            model_type="qwen4_exp",
            text_model_type=text_type,
            vocab_size=_positive(text.get("vocab_size"), "vocab_size"),
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rope_theta=float(rope.get("rope_theta", 10_000_000.0)),
            rope_sections=sections,
            mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
            max_position_embeddings=_positive(
                text.get("max_position_embeddings"), "max_position_embeddings"
            ),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            layer_types=layer_types,
            full_attention_interval=interval,
            hc_count=hc_count,
            hc_lowrank=_positive(text.get("hc_lowrank"), "hc_lowrank"),
            linear_num_key_heads=_positive(
                text.get("linear_num_key_heads"), "linear_num_key_heads"
            ),
            linear_num_value_heads=_positive(
                text.get("linear_num_value_heads"), "linear_num_value_heads"
            ),
            linear_key_head_dim=linear_key_dim,
            linear_value_head_dim=linear_value_dim,
            linear_conv_kernel_dim=_positive(
                text.get("linear_conv_kernel_dim"), "linear_conv_kernel_dim"
            ),
            num_experts=_positive(text.get("num_experts"), "num_experts"),
            num_experts_per_tok=_positive(text.get("num_experts_per_tok"), "num_experts_per_tok"),
            moe_intermediate_size=_positive(
                text.get("moe_intermediate_size"), "moe_intermediate_size"
            ),
            shared_expert_intermediate_size=_positive(
                text.get("shared_expert_intermediate_size"),
                "shared_expert_intermediate_size",
            ),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            output_gate_type=output_gate_type,
            indexer_n_heads=_positive(text.get("indexer_n_heads"), "indexer_n_heads"),
            indexer_kv_heads=indexer_kv_heads,
            indexer_head_dim=_positive(text.get("indexer_head_dim"), "indexer_head_dim"),
            indexer_budget=budget,
            indexer_compress_ratio=compress,
            ple_layer_ids=ple_layers,
            ple_embed_dim=ple_embed,
            ple_conv_kernel_size=_positive(
                text.get("ple_conv_kernel_size"), "ple_conv_kernel_size"
            ),
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            ngram_vocab_size_base=_positive(
                text.get("ngram_vocab_size_base"), "ngram_vocab_size_base"
            ),
            split_ngram_parts=_positive(text.get("split_ngram_parts"), "split_ngram_parts"),
            make_ngram_vocab_size_divisible_by=_positive(
                text.get("make_ngram_vocab_size_divisible_by", 128),
                "make_ngram_vocab_size_divisible_by",
            ),
            seed=int(text.get("seed", 1234) or 1234),
            mtp_num_hidden_layers=mtp_count,
            mtp_use_dedicated_embeddings=bool(text.get("mtp_use_dedicated_embeddings", False)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            eos_token_ids=_eos_ids(text.get("eos_token_id", outer.get("eos_token_id"))),
            image_token_id=(
                None if outer.get("image_token_id") is None else int(outer["image_token_id"])
            ),
            video_token_id=(
                None if outer.get("video_token_id") is None else int(outer["video_token_id"])
            ),
            vision_start_token_id=(
                None
                if outer.get("vision_start_token_id") is None
                else int(outer["vision_start_token_id"])
            ),
            vision_end_token_id=(
                None
                if outer.get("vision_end_token_id") is None
                else int(outer["vision_end_token_id"])
            ),
            vision=vision,
        )


@dataclass(frozen=True)
class Glm5NextConfig:
    family: str
    model_type: str
    text_model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel_size: int
    kda_gate_lower_bound: float
    q_lora_rank: int
    kv_lora_rank: int
    num_attention_heads: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    indexer_types: tuple[str, ...]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    swiglu_limit: float
    scoring_func: str
    norm_topk_prob: bool
    num_nextn_predict_layers: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    image_token_id: int | None
    video_token_id: int | None
    image_start_token_id: int | None
    image_end_token_id: int | None
    video_start_token_id: int | None
    video_end_token_id: int | None
    vision: VisionTowerConfig | None

    @classmethod
    def from_hf_config(cls, outer: Mapping[str, object]) -> Glm5NextConfig:
        outer_type = str(outer.get("model_type", ""))
        text = _object(outer.get("text_config", outer), "text_config")
        text_type = str(text.get("model_type", outer_type))
        if outer_type not in {"glm5_next", "glm5_next_text"} or text_type != "glm5_next_text":
            raise ValueError(f"expected glm5_next/glm5_next_text, got {outer_type!r}/{text_type!r}")
        layers = _positive(text.get("num_hidden_layers"), "num_hidden_layers")
        layer_types = _strings(text.get("layer_types"), "layer_types")
        mlp_types = _strings(text.get("mlp_layer_types"), "mlp_layer_types")
        if len(layer_types) != layers or set(layer_types) - {
            "linear_attention",
            "deepseek_sparse_attention",
        }:
            raise ValueError("GLM-5-Next layer_types do not describe every decoder layer")
        if len(mlp_types) != layers or set(mlp_types) - {"dense", "sparse"}:
            raise ValueError("GLM-5-Next mlp_layer_types do not describe every decoder layer")
        linear = _object(text.get("linear_attn_config"), "linear_attn_config")
        kda_layers = set(_integers(linear.get("kda_layers", ()), "kda_layers"))
        full_layers = set(_integers(linear.get("full_attn_layers", ()), "full_attn_layers"))
        expected_kda = {i for i, value in enumerate(layer_types) if value == "linear_attention"}
        expected_full = set(range(layers)) - expected_kda
        if kda_layers != expected_kda or full_layers != expected_full:
            raise ValueError("GLM-5-Next KDA/sparse-attention schedules disagree")
        if not bool(text.get("mhc", False)) or not bool(text.get("mla_use_nope", False)):
            raise ValueError("GLM-5-Next requires mHC and NoPE MLA")
        if int(text.get("qk_rope_head_dim", 0) or 0) != 0:
            raise ValueError("GLM-5-Next sparse MLA must not use RoPE channels")
        pool = _positive(text.get("index_kpool"), "index_kpool")
        topk = _positive(text.get("index_topk"), "index_topk")
        if topk % pool:
            raise ValueError("GLM-5-Next DSA top-k must divide its k-pool width")
        indexer_types = _strings(text.get("indexer_types"), "indexer_types")
        if len(indexer_types) != layers or set(indexer_types) - {"full", "shared"}:
            raise ValueError("GLM-5-Next indexer schedule must describe every layer")
        if any(
            indexer_types[index] != "full"
            for index, value in enumerate(layer_types)
            if value == "deepseek_sparse_attention"
        ):
            raise ValueError("GLM-5-Next sparse layers require their published full indexers")
        if (
            str(text.get("hidden_act", "silu")) != "silu"
            or bool(text.get("attention_bias", False))
            or str(text.get("scoring_func", "sigmoid")) != "sigmoid"
            or int(text.get("n_group", 1)) != 1
            or int(text.get("topk_group", 1)) != 1
        ):
            raise ValueError("unsupported GLM-5-Next activation/router semantics")
        hidden = _positive(text.get("hidden_size"), "hidden_size")
        vision_raw = outer.get("vision_config")
        vision = (
            None
            if vision_raw is None
            else VisionTowerConfig.from_mapping(_object(vision_raw, "vision_config"))
        )
        if vision is not None and vision.out_hidden_size != hidden:
            raise ValueError("GLM-5-Next vision output width must equal text hidden_size")
        if vision is not None and (
            vision.hidden_act != "silu"
            or not vision.attention_bias
            or vision.projection_intermediate_size is None
            or vision.swiglu_limit <= 0.0
        ):
            raise ValueError("unsupported GLM-5-Next vision-tower semantics")
        return cls(
            family="glm5_next",
            model_type="glm5_next",
            text_model_type=text_type,
            vocab_size=_positive(text.get("vocab_size"), "vocab_size"),
            hidden_size=hidden,
            intermediate_size=_positive(text.get("intermediate_size"), "intermediate_size"),
            num_hidden_layers=layers,
            max_position_embeddings=_positive(
                text.get("max_position_embeddings"), "max_position_embeddings"
            ),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-5)),
            layer_types=layer_types,
            mlp_layer_types=mlp_types,
            hc_mult=_positive(text.get("hc_mult"), "hc_mult"),
            hc_eps=float(text.get("hc_eps", 1e-6)),
            hc_sinkhorn_iters=_positive(text.get("hc_sinkhorn_iters"), "hc_sinkhorn_iters"),
            kda_num_heads=_positive(linear.get("num_heads"), "linear_attn.num_heads"),
            kda_head_dim=_positive(linear.get("head_dim"), "linear_attn.head_dim"),
            kda_conv_kernel_size=_positive(
                linear.get("short_conv_kernel_size"),
                "linear_attn.short_conv_kernel_size",
            ),
            kda_gate_lower_bound=float(linear.get("gate_lower_bound", -5.0)),
            q_lora_rank=_positive(text.get("q_lora_rank"), "q_lora_rank"),
            kv_lora_rank=_positive(text.get("kv_lora_rank"), "kv_lora_rank"),
            num_attention_heads=_positive(text.get("num_attention_heads"), "num_attention_heads"),
            qk_nope_head_dim=_positive(text.get("qk_nope_head_dim"), "qk_nope_head_dim"),
            v_head_dim=_positive(text.get("v_head_dim"), "v_head_dim"),
            index_n_heads=_positive(text.get("index_n_heads"), "index_n_heads"),
            index_head_dim=_positive(text.get("index_head_dim"), "index_head_dim"),
            index_topk=topk,
            index_kpool=pool,
            index_kpool_always_select_tail=bool(text.get("index_kpool_always_select_tail", False)),
            indexer_types=indexer_types,
            num_experts=_positive(text.get("n_routed_experts"), "n_routed_experts"),
            num_experts_per_tok=_positive(text.get("num_experts_per_tok"), "num_experts_per_tok"),
            moe_intermediate_size=_positive(
                text.get("moe_intermediate_size"), "moe_intermediate_size"
            ),
            n_shared_experts=_positive(text.get("n_shared_experts"), "n_shared_experts"),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(text.get("swiglu_limit", 0.0)),
            scoring_func=str(text.get("scoring_func", "sigmoid")),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            num_nextn_predict_layers=int(text.get("num_nextn_predict_layers", 0) or 0),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            eos_token_ids=_eos_ids(text.get("eos_token_id", outer.get("eos_token_id"))),
            image_token_id=(
                None if outer.get("image_token_id") is None else int(outer["image_token_id"])
            ),
            video_token_id=(
                None if outer.get("video_token_id") is None else int(outer["video_token_id"])
            ),
            image_start_token_id=(
                None
                if outer.get("image_start_token_id") is None
                else int(outer["image_start_token_id"])
            ),
            image_end_token_id=(
                None
                if outer.get("image_end_token_id") is None
                else int(outer["image_end_token_id"])
            ),
            video_start_token_id=(
                None
                if outer.get("video_start_token_id") is None
                else int(outer["video_start_token_id"])
            ),
            video_end_token_id=(
                None
                if outer.get("video_end_token_id") is None
                else int(outer["video_end_token_id"])
            ),
            vision=vision,
        )


FlashNextConfig: TypeAlias = Qwen4ExpConfig | Glm5NextConfig


def parse_flash_next_config(config: Mapping[str, object]) -> FlashNextConfig:
    model_type = str(config.get("model_type", ""))
    text = config.get("text_config")
    text_type = str(text.get("model_type", "")) if isinstance(text, Mapping) else ""
    identity = text_type or model_type
    if identity == "qwen4_exp_text" or model_type == "qwen4_exp":
        return Qwen4ExpConfig.from_hf_config(config)
    if identity == "glm5_next_text" or model_type == "glm5_next":
        return Glm5NextConfig.from_hf_config(config)
    raise ValueError(f"unsupported Flash-Next architecture: {model_type!r}/{text_type!r}")


__all__ = [
    "FlashNextConfig",
    "Glm5NextConfig",
    "Qwen4ExpConfig",
    "VisionTowerConfig",
    "parse_flash_next_config",
]
