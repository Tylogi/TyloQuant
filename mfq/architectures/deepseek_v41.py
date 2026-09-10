"""Strict architecture contract for DeepSeek-V4.1-Flash.

DeepSeek-V4.1 is a CED/CSA2 model with Engram conditional memory,
Mega-mHC residual streams, native vision, and a three-stage DSpark head.  It
is deliberately represented independently from the older DeepSeek-V4 family;
only architecture-neutral storage and kernel primitives may be shared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive(value: object, name: str) -> int:
    result = int(value or 0)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


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
class DeepseekV41VisionConfig:
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int
    max_image_tokens: int
    min_pixels: int
    max_wh_ratio: float | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> DeepseekV41VisionConfig:
        model_type = str(raw.get("model_type", ""))
        if model_type != "deepseek_v41_vision":
            raise ValueError(f"expected deepseek_v41_vision, got {model_type!r}")
        ratio = _positive(raw.get("downsample_ratio"), "vision.downsample_ratio")
        heads = _positive(raw.get("num_attention_heads"), "vision.num_attention_heads")
        hidden = _positive(raw.get("hidden_size"), "vision.hidden_size")
        if hidden % heads:
            raise ValueError("DeepSeek-V4.1 vision hidden size must divide its heads")
        max_wh_ratio = raw.get("max_wh_ratio")
        return cls(
            model_type=model_type,
            num_hidden_layers=_positive(
                raw.get("num_hidden_layers"), "vision.num_hidden_layers"
            ),
            hidden_size=hidden,
            num_attention_heads=heads,
            intermediate_size=_positive(
                raw.get("intermediate_size"), "vision.intermediate_size"
            ),
            patch_size=_positive(raw.get("patch_size"), "vision.patch_size"),
            rope_theta=float(raw.get("rope_theta", 10_000.0)),
            downsample_ratio=ratio,
            max_image_tokens=_positive(
                raw.get("max_image_tokens"), "vision.max_image_tokens"
            ),
            min_pixels=_positive(raw.get("min_pixels"), "vision.min_pixels"),
            max_wh_ratio=None if max_wh_ratio is None else float(max_wh_ratio),
        )


@dataclass(frozen=True)
class DeepseekV41Config:
    family: str
    model_type: str
    text_model_type: str
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    causal_encoder_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    rope_factor: float
    original_max_position_embeddings: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float
    sliding_window: int
    compress_ratios: tuple[int, ...]
    compress_rope_theta: float
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    engram_compressed_vocab_size: int
    num_nextn_predict_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_num_experts_per_tok: int
    dense_weight_block_size: tuple[int, int]
    dense_scale_format: str
    expert_dtype: str
    image_token_id: int
    eos_token_ids: tuple[int, ...]
    vision: DeepseekV41VisionConfig

    @classmethod
    def from_hf_config(cls, outer: Mapping[str, object]) -> DeepseekV41Config:
        outer_type = str(outer.get("model_type", ""))
        text = _object(outer.get("text_config"), "text_config")
        text_type = str(text.get("model_type", ""))
        if outer_type != "deepseek_v41" or text_type != "deepseek_v41_text":
            raise ValueError(
                f"expected deepseek_v41/deepseek_v41_text, got "
                f"{outer_type!r}/{text_type!r}"
            )

        layers = _positive(text.get("num_hidden_layers"), "num_hidden_layers")
        mtp_layers = _positive(
            text.get("num_nextn_predict_layers"), "num_nextn_predict_layers"
        )
        ratios = _integers(text.get("compress_ratios"), "compress_ratios")
        if len(ratios) != layers + mtp_layers or any(value not in {0, 1, 2} for value in ratios):
            raise ValueError(
                "DeepSeek-V4.1 compression schedule must cover every backbone and DSpark layer"
            )
        backbone_ratios = ratios[:layers]
        try:
            encoder_layers = backbone_ratios.index(1)
        except ValueError as error:
            raise ValueError("DeepSeek-V4.1 CED schedule has no decoder segment") from error
        if (
            encoder_layers <= 0
            or any(value == 1 for value in backbone_ratios[:encoder_layers])
            or any(value != 1 for value in backbone_ratios[encoder_layers:])
            or any(ratios[layers:])
        ):
            raise ValueError("DeepSeek-V4.1 CED/DSpark compression schedule is inconsistent")

        kv_sources = _integers(text.get("kv_source_layer_ids"), "kv_source_layer_ids")
        index_sources = _integers(
            text.get("index_source_layer_ids"), "index_source_layer_ids"
        )
        candidate_source = int(text.get("candidate_source_layer_id", -1))
        if (
            not kv_sources
            or not set(kv_sources).issubset(index_sources)
            or any(layer < 0 or layer >= layers for layer in index_sources)
            or candidate_source not in index_sources
        ):
            raise ValueError("DeepSeek-V4.1 CSA2 source-layer schedule is inconsistent")
        active_ratio = 0
        kv_source_set = set(kv_sources)
        for layer, ratio in enumerate(backbone_ratios):
            if layer in kv_source_set:
                if ratio <= 0:
                    raise ValueError(
                        "DeepSeek-V4.1 KV source has no compressed stream"
                    )
                active_ratio = ratio
            if ratio > 0 and active_ratio != ratio:
                raise ValueError(
                    "DeepSeek-V4.1 CSA2 consumer has no compatible KV source"
                )

        engram_layers = _integers(text.get("engram_layer_ids"), "engram_layer_ids")
        engram_rows = _integers(
            text.get("engram_num_embeddings"), "engram_num_embeddings"
        )
        if len(engram_layers) != len(engram_rows) or any(
            layer < 0 or layer >= layers for layer in engram_layers
        ):
            raise ValueError("DeepSeek-V4.1 Engram layers and tables disagree")

        dspark_targets = _integers(
            text.get("dspark_target_layer_ids"), "dspark_target_layer_ids"
        )
        if len(dspark_targets) != mtp_layers or any(
            layer < 0 or layer >= layers for layer in dspark_targets
        ):
            raise ValueError("DeepSeek-V4.1 DSpark target layers disagree with stage count")

        hidden = _positive(text.get("hidden_size"), "hidden_size")
        heads = _positive(text.get("num_attention_heads"), "num_attention_heads")
        kv_heads = _positive(text.get("num_key_value_heads"), "num_key_value_heads")
        head_dim = _positive(text.get("head_dim"), "head_dim")
        rope_dim = _positive(text.get("qk_rope_head_dim"), "qk_rope_head_dim")
        groups = _positive(text.get("o_groups"), "o_groups")
        if heads % kv_heads or heads % groups or rope_dim > head_dim or rope_dim % 2:
            raise ValueError("DeepSeek-V4.1 attention dimensions are inconsistent")
        if (
            str(text.get("hidden_act", "")) != "silu"
            or bool(text.get("attention_bias", False))
            or str(text.get("scoring_func", "")) != "sqrtsoftplus"
            or str(text.get("topk_method", "")) != "noaux_tc"
        ):
            raise ValueError("unsupported DeepSeek-V4.1 activation/router semantics")

        quantization = _object(outer.get("quantization_config"), "quantization_config")
        block_size = _integers(
            quantization.get("weight_block_size"), "quantization.weight_block_size"
        )
        if (
            str(quantization.get("quant_method", "")) != "fp8"
            or block_size != (32, 32)
            or str(quantization.get("scale_fmt", "")) != "ue8m0"
            or str(quantization.get("expert_dtype", "")) != "fp4"
        ):
            raise ValueError("unsupported DeepSeek-V4.1 native weight encoding")

        rope = _object(text.get("rope_scaling"), "rope_scaling")
        if str(rope.get("rope_type", "")) != "yarn":
            raise ValueError("DeepSeek-V4.1 requires YaRN RoPE scaling")
        vision = DeepseekV41VisionConfig.from_mapping(
            _object(outer.get("vision_config"), "vision_config")
        )
        return cls(
            family="deepseek_v41",
            model_type=outer_type,
            text_model_type=text_type,
            vocab_size=_positive(text.get("vocab_size"), "vocab_size"),
            hidden_size=hidden,
            moe_intermediate_size=_positive(
                text.get("moe_intermediate_size"), "moe_intermediate_size"
            ),
            num_hidden_layers=layers,
            causal_encoder_layers=encoder_layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            qk_rope_head_dim=rope_dim,
            q_lora_rank=_positive(text.get("q_lora_rank"), "q_lora_rank"),
            o_lora_rank=_positive(text.get("o_lora_rank"), "o_lora_rank"),
            o_groups=groups,
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-20)),
            max_position_embeddings=_positive(
                text.get("max_position_embeddings"), "max_position_embeddings"
            ),
            rope_theta=float(text.get("rope_theta", 10_000.0)),
            rope_factor=float(rope.get("factor", 1.0)),
            original_max_position_embeddings=_positive(
                rope.get("original_max_position_embeddings"),
                "rope_scaling.original_max_position_embeddings",
            ),
            n_routed_experts=_positive(text.get("n_routed_experts"), "n_routed_experts"),
            n_shared_experts=_positive(text.get("n_shared_experts"), "n_shared_experts"),
            num_experts_per_tok=_positive(
                text.get("num_experts_per_tok"), "num_experts_per_tok"
            ),
            scoring_func=str(text.get("scoring_func")),
            norm_topk_prob=bool(text.get("norm_topk_prob", False)),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(text.get("swiglu_limit", 0.0)),
            sliding_window=_positive(text.get("sliding_window"), "sliding_window"),
            compress_ratios=ratios,
            compress_rope_theta=float(text.get("compress_rope_theta", 160_000.0)),
            kv_source_layer_ids=kv_sources,
            index_source_layer_ids=index_sources,
            index_n_heads=_positive(text.get("index_n_heads"), "index_n_heads"),
            index_head_dim=_positive(text.get("index_head_dim"), "index_head_dim"),
            index_topk=_positive(text.get("index_topk"), "index_topk"),
            candidate_source_layer_id=candidate_source,
            candidate_topk_blocks=_positive(
                text.get("candidate_topk_blocks"), "candidate_topk_blocks"
            ),
            candidate_block_size=_positive(
                text.get("candidate_block_size"), "candidate_block_size"
            ),
            hc_mult=_positive(text.get("hc_mult"), "hc_mult"),
            hc_sinkhorn_iters=_positive(
                text.get("hc_sinkhorn_iters"), "hc_sinkhorn_iters"
            ),
            hc_eps=float(text.get("hc_eps", 1e-6)),
            engram_layer_ids=engram_layers,
            engram_num_embeddings=engram_rows,
            engram_max_ngram_size=_positive(
                text.get("engram_max_ngram_size"), "engram_max_ngram_size"
            ),
            engram_vocab_size=_positive(text.get("engram_vocab_size"), "engram_vocab_size"),
            engram_n_heads=_positive(text.get("engram_n_heads"), "engram_n_heads"),
            engram_head_dim=_positive(text.get("engram_head_dim"), "engram_head_dim"),
            engram_pad_token_id=int(text.get("engram_pad_token_id", 0)),
            engram_compressed_vocab_size=_positive(
                text.get("engram_compressed_vocab_size"), "engram_compressed_vocab_size"
            ),
            num_nextn_predict_layers=mtp_layers,
            dspark_block_size=_positive(text.get("dspark_block_size"), "dspark_block_size"),
            dspark_noise_token_id=int(text.get("dspark_noise_token_id", 0)),
            dspark_target_layer_ids=dspark_targets,
            dspark_markov_rank=_positive(
                text.get("dspark_markov_rank"), "dspark_markov_rank"
            ),
            dspark_n_routed_experts=_positive(
                text.get("dspark_n_routed_experts"), "dspark_n_routed_experts"
            ),
            dspark_num_experts_per_tok=_positive(
                text.get("dspark_num_experts_per_tok"), "dspark_num_experts_per_tok"
            ),
            dense_weight_block_size=(block_size[0], block_size[1]),
            dense_scale_format=str(quantization.get("scale_fmt")),
            expert_dtype=str(quantization.get("expert_dtype")),
            image_token_id=int(outer.get("image_token_id", -1)),
            eos_token_ids=_eos_ids(outer.get("eos_token_id")),
            vision=vision,
        )


def parse_deepseek_v41_config(config: Mapping[str, object]) -> DeepseekV41Config:
    return DeepseekV41Config.from_hf_config(config)


__all__ = [
    "DeepseekV41Config",
    "DeepseekV41VisionConfig",
    "parse_deepseek_v41_config",
]
