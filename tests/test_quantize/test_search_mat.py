"""Full-evaluation tests for per-tensor NintSpec search in search_mat."""

from __future__ import annotations

import numpy as np
import pytest

from mfq.quantize import search_mat
from mfq.quantize.nint_quant import dequantize, quantize
from mfq.utils.tensor import snr


def _gauss(out: int, inn: int, seed: int = 0, sigma: float = 0.05) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, sigma, size=(out, inn)).astype(np.float32)


def test_best_spec_under_budget_and_decent_snr():
    W = _gauss(64, 5120)
    res = search_mat.search(W, target_bpw=4.51, axis=0)
    assert res.spec.bpw(5120) <= 4.51 + 1e-9
    assert res.snr_db > 21.0


def test_best_spec_in_gs_sweet_region():
    W = _gauss(64, 5120, seed=1)
    res = search_mat.search(W, target_bpw=4.51)
    assert res.spec.groupsize in (16, 24, 32, 48, 64)  # PROFILE_CATALOG


def test_snr_matches_full_recompute():
    """Returned snr_db should match SNR recomputed over all weights without subsampling."""
    W = _gauss(64, 5120, seed=2)
    res = search_mat.search(W, target_bpw=4.6)
    full = snr(W, dequantize(quantize(W, res.spec, axis=0)))
    assert abs(full - res.snr_db) < 1e-6


def test_higher_budget_no_worse():
    W = _gauss(64, 5120, seed=3)
    lo = search_mat.search(W, target_bpw=4.3)
    hi = search_mat.search(W, target_bpw=4.7)
    assert hi.snr_db >= lo.snr_db - 1e-6


def test_rejects_too_low_budget():
    W = _gauss(32, 96)
    with pytest.raises(ValueError):
        # NINT2 is now part of PROFILE_CATALOG and reaches ~2.96 bpw here.
        search_mat.search(W, target_bpw=2.0)


def test_rejects_1d():
    with pytest.raises(ValueError):
        search_mat.search(np.zeros(64, dtype=np.float32), 4.5)


@pytest.mark.parametrize("target,bits", [(5.6, 5), (6.7, 6), (9.2, 8)])
def test_search_can_select_high_bit_profiles(target, bits):
    W = _gauss(32, 512, seed=bits)
    res = search_mat.search(W, target_bpw=target)
    assert res.spec.bits == bits
    assert res.spec.bpw(512) <= target + 1e-9


def test_evaluated_sorted_desc():
    W = _gauss(32, 5120, seed=4)
    res = search_mat.search(W, target_bpw=4.5)
    snrs = [s for _, s, _ in res.evaluated]
    assert snrs == sorted(snrs, reverse=True)


def test_full_eval_no_max_rows_param():
    """Search no longer accepts max_rows; full evaluation is the only behavior."""
    import inspect

    sig = inspect.signature(search_mat.search)
    assert "max_rows" not in sig.parameters
