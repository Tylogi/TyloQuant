"""TPQ 通用算子层。"""

from .api import (
    attention_step,
    create_tensor_parallel,
    gated_activation,
    linear_route_topk,
    packed_moe_topk,
    packed_moe_selected_topk,
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
)

__all__ = [
    "ModelOperatorConfig",
    "OperatorCapability",
    "OperatorRegistry",
    "OperatorRequest",
    "REGISTRY",
    "attention_step",
    "create_tensor_parallel",
    "gated_activation",
    "linear_route_topk",
    "packed_moe_topk",
    "packed_moe_selected_topk",
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
    "FixedMoEPrelude",
    "FixedMoEPreludeSpec",
    "vq_gemv",
    "vq_gemv_packed_list",
]
