from __future__ import annotations

import numpy as np

from bench.dsv4f_mxfp4_sq2_palette_screen import _greedy_catalog


def test_greedy_catalog_preserves_seed_and_improves_partition_coverage() -> None:
    errors = np.asarray(
        [
            [3.0, 2.0, 1.0, 5.0, 5.0],
            [1.0, 2.0, 3.0, 0.5, 5.0],
            [2.0, 1.0, 3.0, 5.0, 0.25],
        ],
        dtype=np.float64,
    )
    selected = _greedy_catalog(errors, seed=np.asarray([0, 1]), size=4)

    assert selected[:2] == [0, 1]
    assert selected[2:] == [2, 4]
    before = errors[:, selected[:2]].min(axis=1).sum()
    after = errors[:, selected].min(axis=1).sum()
    assert after < before
