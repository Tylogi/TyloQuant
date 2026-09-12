"""Source contracts for DeepSeek V4 Flash M3 Ultra fast paths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "cpp_runtime" / "backends" / "metal"
MOE = (METAL / "ops" / "mlx_moe.cpp").read_text(encoding="utf-8")


def test_m3_ultra_dsv4f_mtp_uses_native_mxfp4_smallm_nax() -> None:
    assert "const bool dsv4f_m3_ultra" in MOE
    assert "apple_m3_ultra() && experts == 256" in MOE
    assert "input_width == 4096" in MOE
    assert "output_width == 2048 || output_width == 4096" in MOE
    assert "input_width == 2048 && output_width == 4096" in MOE
    assert "!apple_m5_family() && !dsv4f_m3_ultra" in MOE
    assert "impl_->neuron_len" in MOE
    assert "impl_->out_per_expert" in MOE
