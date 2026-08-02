"""TPQ 通用算子层。"""

from .api import (
    attention_step,
    block_scaled_gemv,
    block_scaled_grouped_gemv,
    create_tensor_parallel,
    gated_activation,
    hyper_connection_post,
    hyper_connection_post_moe,
    hyper_connection_pre_norm,
    linear,
    linear_route_topk,
    packed_moe_topk,
    packed_moe_selected_topk,
    packed_moe_operator_name,
    packed_route_slots,
    resident_moe_topk,
    residual_add3,
    residual_mix,
    route_topk,
    rmsnorm,
    vq_gemv,
    vq_gemv_packed_list,
)
from .hidden import TPHidden, TPPartials, TPResidualBuffer, TPSharded
from .moe import FixedMoEPrelude, FixedMoEPreludeSpec
from .profiling import TPHiddenStageProfiler
from .config import ModelOperatorConfig
from .registry import OperatorRegistry, REGISTRY
from .spec import OperatorCapability, OperatorRequest
from .tensor_parallel import (
    TensorParallelDecodeLayerPlan,
    TensorParallelMoELayerPlan,
    TensorParallelVocab,
    shard_linear_input,
    shard_linear_output,
)

__all__ = [
    "ModelOperatorConfig",
    "OperatorCapability",
    "OperatorRegistry",
    "OperatorRequest",
    "REGISTRY",
    "attention_step",
    "block_scaled_gemv",
    "block_scaled_grouped_gemv",
    "create_tensor_parallel",
    "gated_activation",
    "hyper_connection_post",
    "hyper_connection_post_moe",
    "hyper_connection_pre_norm",
    "linear",
    "linear_route_topk",
    "packed_moe_topk",
    "packed_moe_selected_topk",
    "packed_moe_operator_name",
    "packed_route_slots",
    "resident_moe_topk",
    "residual_add3",
    "residual_mix",
    "route_topk",
    "rmsnorm",
    "TPHidden",
    "TPHiddenStageProfiler",
    "TPPartials",
    "TPResidualBuffer",
    "TPSharded",
    "TensorParallelMoELayerPlan",
    "TensorParallelDecodeLayerPlan",
    "TensorParallelVocab",
    "shard_linear_input",
    "shard_linear_output",
    "FixedMoEPrelude",
    "FixedMoEPreludeSpec",
    "vq_gemv",
    "vq_gemv_packed_list",
]
