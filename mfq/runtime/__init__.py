"""MFQ inference runtimes.

The NumPy reference runtime is always available. Torch/CUDA, TPQ, and
MLX/Metal objects are imported lazily so installing one optional backend does
not require the dependencies of another backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mfq.runtime.dequantize import clear_backends, dequantize, register_backend
from mfq.runtime.ffn import SwiGLUFFN, silu
from mfq.runtime.linear import NintLinear
from mfq.runtime.model import NintModel
from mfq.runtime.tpq import (
    CCCPArtifact,
    TPQArtifact,
    configure_cccp_memory,
    configure_tpq_memory,
    load_cccp_model,
    load_tpq_model,
    open_cccp_artifact,
    open_tpq_artifact,
    run_cccp_chat,
    run_tpq_chat,
)

if TYPE_CHECKING:
    from mfq.runtime.causal_lm import (
        TorchNintCausalLM,
        TorchNintCausalLMConfig,
        TorchNintCausalLMNames,
    )
    from mfq.runtime.minicpmo45 import (
        MiniCPMO45LoadReport,
        TorchMfqMiniCPMO45,
        load_minicpmo45,
    )
    from mfq.runtime.mlx_attention import MlxKVCache, MlxSlidingWindowKVCache
    from mfq.runtime.mlx_causal_lm import (
        MlxCausalLM,
        MlxCausalLMConfig,
        MlxCausalLMNames,
        MlxFullAttentionBlock,
        MlxQwen35LinearAttentionBlock,
    )
    from mfq.runtime.mlx_deepseek_v4 import (
        MlxDeepseekV4,
        MlxDeepseekV4Attention,
        MlxDeepseekV4Config,
        MlxDeepseekV4MoE,
        MlxDeepseekV4Names,
        MlxDeepseekV4PoolState,
    )
    from mfq.runtime.mlx_gemma4 import (
        MlxGemma4,
        MlxGemma4Config,
        MlxGemma4DenseFFN,
        MlxGemma4Layer,
        MlxGemma4MoE,
        MlxGemma4Names,
    )
    from mfq.runtime.mlx_glm_dsa import (
        MlxGlmDsa,
        MlxGlmDsaConfig,
        MlxGlmDsaDenseFFN,
        MlxGlmDsaLayer,
        MlxGlmDsaMoE,
        MlxGlmDsaNames,
    )
    from mfq.runtime.mlx_kimi_k3 import (
        MlxKimiK3,
        MlxKimiK3Config,
        MlxKimiK3Names,
        MlxKimiKDA,
        MlxKimiMLA,
        MlxKimiMoE,
        MlxKimiSiTUFFN,
    )
    from mfq.runtime.mlx_linear import (
        MlxDenseEmbedding,
        MlxDenseLinear,
        MlxLinearGroup,
        MlxMxEmbedding,
        MlxMxLinear,
        MlxNint8ZeroEmbedding,
        MlxNint8ZeroLinear,
        MlxNintEmbedding,
        MlxNintLinear,
        MlxNintModel,
        MlxSwiGLUFFN,
    )
    from mfq.runtime.mlx_moe import (
        MlxRoutedLinear,
        MlxRoutedSiTUFFN,
        MlxRoutedSwiGLUFFN,
    )
    from mfq.runtime.mlx_ops import MlxRMSNorm, MlxRoPE
    from mfq.runtime.mlx_tpq import (
        MlxCccpInt4Embedding,
        MlxCccpInt4Linear,
        MlxCccpPqLinear,
        MlxTpqInt4Embedding,
        MlxTpqInt4Linear,
        MlxTpqPqLinear,
    )
    from mfq.runtime.mlx_vq import MlxVqEmbedding, MlxVqLinear
    from mfq.runtime.tpq_mfq import (
        MfqCccpStore,
        MfqTpqStore,
        NativeCCCPArtifact,
        NativeTPQArtifact,
    )


_TORCH_EXPORTS = {
    "TorchNintCausalLM",
    "TorchNintCausalLMConfig",
    "TorchNintCausalLMNames",
}
_MINICPMO45_EXPORTS = {
    "MiniCPMO45LoadReport",
    "TorchMfqMiniCPMO45",
    "load_minicpmo45",
}
_TPQ_MFQ_EXPORTS = {
    "MfqCccpStore",
    "MfqTpqStore",
    "NativeCCCPArtifact",
    "NativeTPQArtifact",
    "install_mfq_cccp_store",
    "install_mfq_tpq_store",
}
_MLX_EXPORTS = {
    "MlxCccpInt4Embedding",
    "MlxCccpInt4Linear",
    "MlxCccpPqLinear",
    "MlxTpqInt4Embedding",
    "MlxTpqInt4Linear",
    "MlxTpqPqLinear",
    "MlxDenseEmbedding",
    "MlxDenseLinear",
    "MlxLinearGroup",
    "MlxMxEmbedding",
    "MlxMxLinear",
    "MlxNint8ZeroEmbedding",
    "MlxNint8ZeroLinear",
    "MlxNintEmbedding",
    "MlxNintLinear",
    "MlxNintModel",
    "MlxRMSNorm",
    "MlxRoPE",
    "MlxSwiGLUFFN",
    "MlxVqEmbedding",
    "MlxVqLinear",
}
_MLX_OP_EXPORTS = {"MlxRMSNorm", "MlxRoPE"}
_MLX_ATTENTION_EXPORTS = {"MlxKVCache", "MlxSlidingWindowKVCache"}
_MLX_CAUSAL_LM_EXPORTS = {
    "MlxCausalLM",
    "MlxCausalLMConfig",
    "MlxCausalLMNames",
    "MlxFullAttentionBlock",
    "MlxQwen35LinearAttentionBlock",
}
_MLX_TPQ_EXPORTS = {
    "MlxCccpInt4Embedding",
    "MlxCccpInt4Linear",
    "MlxCccpPqLinear",
    "MlxTpqInt4Embedding",
    "MlxTpqInt4Linear",
    "MlxTpqPqLinear",
}
_MLX_DEEPSEEK_V4_EXPORTS = {
    "MlxDeepseekV4",
    "MlxDeepseekV4Attention",
    "MlxDeepseekV4Config",
    "MlxDeepseekV4MoE",
    "MlxDeepseekV4Names",
    "MlxDeepseekV4PoolState",
}
_MLX_GEMMA4_EXPORTS = {
    "MlxGemma4",
    "MlxGemma4Config",
    "MlxGemma4DenseFFN",
    "MlxGemma4Layer",
    "MlxGemma4MoE",
    "MlxGemma4Names",
}
_MLX_GLM_DSA_EXPORTS = {
    "MlxGlmDsa",
    "MlxGlmDsaConfig",
    "MlxGlmDsaDenseFFN",
    "MlxGlmDsaLayer",
    "MlxGlmDsaMoE",
    "MlxGlmDsaNames",
}
_MLX_KIMI_EXPORTS = {
    "MlxKimiK3",
    "MlxKimiK3Config",
    "MlxKimiK3Names",
    "MlxKimiKDA",
    "MlxKimiMLA",
    "MlxKimiMoE",
    "MlxKimiSiTUFFN",
}
_MLX_MOE_EXPORTS = {
    "MlxRoutedLinear",
    "MlxRoutedSiTUFFN",
    "MlxRoutedSwiGLUFFN",
}


def __getattr__(name: str):
    if name in _TPQ_MFQ_EXPORTS:
        from mfq.runtime import tpq_mfq

        value = getattr(tpq_mfq, name)
    elif name in _TORCH_EXPORTS:
        from mfq.runtime import causal_lm

        value = getattr(causal_lm, name)
    elif name in _MINICPMO45_EXPORTS:
        from mfq.runtime import minicpmo45

        value = getattr(minicpmo45, name)
    elif name in _MLX_OP_EXPORTS:
        from mfq.runtime import mlx_ops

        value = getattr(mlx_ops, name)
    elif name in _MLX_ATTENTION_EXPORTS:
        from mfq.runtime import mlx_attention

        value = getattr(mlx_attention, name)
    elif name in _MLX_CAUSAL_LM_EXPORTS:
        from mfq.runtime import mlx_causal_lm

        value = getattr(mlx_causal_lm, name)
    elif name in _MLX_TPQ_EXPORTS:
        from mfq.runtime import mlx_tpq

        value = getattr(mlx_tpq, name)
    elif name in _MLX_DEEPSEEK_V4_EXPORTS:
        from mfq.runtime import mlx_deepseek_v4

        value = getattr(mlx_deepseek_v4, name)
    elif name in _MLX_GEMMA4_EXPORTS:
        from mfq.runtime import mlx_gemma4

        value = getattr(mlx_gemma4, name)
    elif name in _MLX_GLM_DSA_EXPORTS:
        from mfq.runtime import mlx_glm_dsa

        value = getattr(mlx_glm_dsa, name)
    elif name in _MLX_KIMI_EXPORTS:
        from mfq.runtime import mlx_kimi_k3

        value = getattr(mlx_kimi_k3, name)
    elif name in _MLX_MOE_EXPORTS:
        from mfq.runtime import mlx_moe

        value = getattr(mlx_moe, name)
    elif name in _MLX_EXPORTS:
        from mfq.runtime import mlx_linear

        value = getattr(mlx_linear, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "TPQArtifact",
    "NativeTPQArtifact",
    "MfqTpqStore",
    "configure_tpq_memory",
    "install_mfq_tpq_store",
    "load_tpq_model",
    "open_tpq_artifact",
    "run_tpq_chat",
    # Legacy public spellings remain aliases for existing integrations.
    "CCCPArtifact",
    "NativeCCCPArtifact",
    "MfqCccpStore",
    "configure_cccp_memory",
    "install_mfq_cccp_store",
    "load_cccp_model",
    "open_cccp_artifact",
    "run_cccp_chat",
    "NintModel",
    "NintLinear",
    "SwiGLUFFN",
    "silu",
    "dequantize",
    "register_backend",
    "clear_backends",
    "TorchNintCausalLM",
    "TorchNintCausalLMConfig",
    "TorchNintCausalLMNames",
    "MiniCPMO45LoadReport",
    "TorchMfqMiniCPMO45",
    "load_minicpmo45",
    "MlxCccpInt4Embedding",
    "MlxCccpInt4Linear",
    "MlxCccpPqLinear",
    "MlxTpqInt4Embedding",
    "MlxTpqInt4Linear",
    "MlxTpqPqLinear",
    "MlxDenseEmbedding",
    "MlxDenseLinear",
    "MlxDeepseekV4",
    "MlxDeepseekV4Attention",
    "MlxDeepseekV4Config",
    "MlxDeepseekV4MoE",
    "MlxDeepseekV4Names",
    "MlxDeepseekV4PoolState",
    "MlxCausalLM",
    "MlxCausalLMConfig",
    "MlxCausalLMNames",
    "MlxFullAttentionBlock",
    "MlxGemma4",
    "MlxGemma4Config",
    "MlxGemma4DenseFFN",
    "MlxGemma4Layer",
    "MlxGemma4MoE",
    "MlxGemma4Names",
    "MlxGlmDsa",
    "MlxGlmDsaConfig",
    "MlxGlmDsaDenseFFN",
    "MlxGlmDsaLayer",
    "MlxGlmDsaMoE",
    "MlxGlmDsaNames",
    "MlxQwen35LinearAttentionBlock",
    "MlxKVCache",
    "MlxKimiK3",
    "MlxKimiK3Config",
    "MlxKimiK3Names",
    "MlxKimiKDA",
    "MlxKimiMLA",
    "MlxKimiMoE",
    "MlxKimiSiTUFFN",
    "MlxLinearGroup",
    "MlxMxEmbedding",
    "MlxMxLinear",
    "MlxNint8ZeroEmbedding",
    "MlxNint8ZeroLinear",
    "MlxNintEmbedding",
    "MlxNintLinear",
    "MlxNintModel",
    "MlxRMSNorm",
    "MlxRoPE",
    "MlxRoutedLinear",
    "MlxRoutedSiTUFFN",
    "MlxRoutedSwiGLUFFN",
    "MlxSwiGLUFFN",
    "MlxSlidingWindowKVCache",
    "MlxVqEmbedding",
    "MlxVqLinear",
]
