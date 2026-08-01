from __future__ import annotations

from mfq.quantize.v4f_upgrade import _profile_routed_bytes
from mfq.tools.v4f_ud_layer_upgrade import (
    DEFAULT_FAMILY_SNR_DB,
    HIGH_FAMILIES,
    solve_ud_layer_upgrade,
)


class _Marks:
    def mark(self, _layer: int, expert: int) -> str:
        if expert == 0:
            return "V"
        if expert < 128:
            return "v"
        return "w"


def test_ud_allocator_is_monotonic_and_enforces_layer_constraints() -> None:
    base = {}
    energy = {}
    for projection in ("gate_up", "down"):
        for layer in range(43):
            families = ["NEPQ0-S"] * 256
            families[1] = "NVQ2J"
            families[2] = "NINT4"
            families[0] = "NINT8"
            base[(projection, layer)] = tuple(families)
    for layer in range(43):
        for expert in range(256):
            energy[(layer, expert)] = {
                "gate_up": float(257 - expert),
                "down": float(257 - expert),
            }
    routed_limit = _profile_routed_bytes(base) + 2_200_000_000
    final, diagnostics = solve_ud_layer_upgrade(
        base,
        energy,
        _Marks(),
        routed_limit=routed_limit,
        family_snr_db=DEFAULT_FAMILY_SNR_DB,
        v_multipliers=(1.0, 4.0),
    )

    assert _profile_routed_bytes(final) <= routed_limit
    assert diagnostics["uniform_gate_up_high_precision_quota"] >= 3
    assert set(diagnostics["down_special_high_precision_counts"].values()) == {
        256
    }
    order = {
        "NEPQ0-S": 0,
        "NVQ2J": 1,
        "NINT4": 2,
        "NINT5": 3,
        "NINT6": 4,
        "NINT8": 5,
    }
    for key, families in final.items():
        for expert, family in enumerate(families):
            assert order[family] >= order[base[key][expert]]
            if expert == 0:
                assert family == "NINT8"
            if expert >= 128:
                assert family in {
                    base[key][expert],
                    "NINT4",
                }
    for count in diagnostics["gate_up_high_precision_counts"].values():
        assert count >= diagnostics["uniform_gate_up_high_precision_quota"]
    assert all(
        family in HIGH_FAMILIES
        for layer in (26, 42)
        for family in final[("down", layer)]
    )
