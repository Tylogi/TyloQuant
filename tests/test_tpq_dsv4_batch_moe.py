from types import MethodType, SimpleNamespace

import torch

from mfq._vendor.tpq import dsv4model


class _FixedResultPool:
    """Minimal packed pool whose public result is overwritten every call."""

    def __init__(self, width: int):
        self.result = torch.empty(1, width)

    def run(self, _layer, value, _ids, _weights, **_kwargs):
        self.result.copy_(value * 2)
        return self.result


def test_dsv4_batched_moe_owns_each_fixed_workspace_result(monkeypatch):
    width = 4
    model = object.__new__(dsv4model.DSV4TPQModel)
    model.cfg = {
        "moe_inter": 2,
        "swiglu_limit": 0.0,
        "situ_beta": 4.0,
        "situ_linear_beta": -1.0,
        "top_k": 1,
        "norm_topk_prob": True,
        "routed_scaling": 1.0,
    }
    model.operator_config = SimpleNamespace(expert_activation="swiglu")
    model.store = SimpleNamespace(
        man=SimpleNamespace(projection_vq=True),
    )
    model.pool = _FixedResultPool(width)
    model._profile_enabled = False
    model._cpu_resident_experts = {}
    model._cpu_fused_resident_moe = {}
    model._cpu_moe_layers = {}
    model._packed_device_pool = True
    model._packed_full_gpu = True
    model._tp_shared_mlp = None
    model.layer = MethodType(lambda _self, _layer: {}, model)
    model._cfg_obj = MethodType(
        lambda _self: SimpleNamespace(
            top_k=1,
            norm_topk_prob=True,
            routed_scaling=1.0,
        ),
        model,
    )
    model._route_tpq = MethodType(
        lambda _self, x, _weights, _cfg, _ids, _layer: (
            torch.ones(x.shape[0], 1),
            torch.zeros(x.shape[0], 1, dtype=torch.long),
        ),
        model,
    )
    monkeypatch.setattr(
        dsv4model,
        "_shared_expert_mlp_tpq",
        lambda x, _weights, _limit: torch.zeros_like(x),
    )

    values = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]
    )
    output = model._moe(
        values,
        layer=0,
        ids=torch.zeros(1, 3, dtype=torch.long),
    )

    torch.testing.assert_close(output, values * 2)
