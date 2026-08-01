from __future__ import annotations

from mfq.quantize.v4f_upgrade import _profile_routed_bytes
from mfq.tools.v4f_sensitivity_overlay import (
    DEFAULT_FAMILY_SNR_DB,
    LayerBpwConstraint,
    _layer_effective_bpw,
    _solve_robust_profiles,
)


class _Marks:
    def mark(self, _layer: int, expert: int) -> str:
        if expert == 0:
            return "V"
        if expert == 1:
            return "v"
        return "w"


class _AllWMarks:
    def mark(self, _layer: int, _expert: int) -> str:
        return "w"


def test_robust_allocator_enforces_marks_and_exact_budget() -> None:
    base = {}
    for projection in ("gate_up", "down"):
        families = ["NVQ2J"] * 256
        families[2:102] = ["NINT4"] * 100
        base[(projection, 0)] = tuple(families)
    energy = {
        (0, expert): {
            "gate_up": float(257 - expert),
            "down": float((257 - expert) ** 2),
        }
        for expert in range(256)
    }
    routed_limit = _profile_routed_bytes(base)

    final, diagnostics = _solve_robust_profiles(
        base,
        energy,
        _Marks(),
        routed_limit=routed_limit,
        family_snr_db=DEFAULT_FAMILY_SNR_DB,
        v_multipliers=(1.0, 4.0),
    )

    assert _profile_routed_bytes(final) <= routed_limit
    for projection in ("gate_up", "down"):
        assert final[(projection, 0)][0] == "NINT8"
        assert final[(projection, 0)][1] in {
            "NVQ2J",
            "NINT4",
            "NINT5",
            "NINT6",
            "NINT8",
        }
        assert set(final[(projection, 0)][2:]) <= {"NVQ2J", "NINT4"}
    assert diagnostics["solver"] == "scipy.optimize.milp/highs"
    assert len(diagnostics["final_to_baseline_ratios"]) == 2


def test_robust_allocator_enforces_relative_layer_bpw() -> None:
    base = {}
    for projection in ("gate_up", "down"):
        for layer in range(2):
            families = ["NVQ2J"] * 256
            if projection == "gate_up" and layer == 0:
                families[:64] = ["NINT4"] * 64
            if projection == "down" and layer == 0:
                families[:64] = ["NINT4"] * 64
            base[(projection, layer)] = tuple(families)
    energy = {
        (layer, expert): {
            "gate_up": 1.0,
            "down": 1.0,
        }
        for layer in range(2)
        for expert in range(256)
    }
    routed_limit = _profile_routed_bytes(base)
    policy = (
        LayerBpwConstraint(
            projection="gate_up",
            high_layers=(1,),
            normal_dtype="LOW",
            high_dtype="HIGH",
            minimum_delta_bpw=0.3,
        ),
    )

    final, diagnostics = _solve_robust_profiles(
        base,
        energy,
        _AllWMarks(),
        routed_limit=routed_limit,
        family_snr_db=DEFAULT_FAMILY_SNR_DB,
        v_multipliers=(1.0,),
        layer_bpw_constraints=policy,
    )

    high = _layer_effective_bpw(final, "gate_up", 1)
    normal = _layer_effective_bpw(final, "gate_up", 0)
    assert high - normal >= 0.3 - 1e-6
    layer_diagnostics = diagnostics["layer_bpw"]
    assert len(layer_diagnostics) == 1
    assert (
        layer_diagnostics[0]["minimum_observed_delta_bpw"]
        >= 0.3 - 1e-6
    )
