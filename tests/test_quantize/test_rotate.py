"""Basic Hadamard-matrix properties: power-of-two n and orthonormality."""

from __future__ import annotations

import numpy as np
import pytest

from mfq.quantize.rotate import hadamard_matrix


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_hadamard_orthonormal(n):
    h = hadamard_matrix(n)
    assert h.shape == (n, n)
    np.testing.assert_allclose(h @ h.T, np.eye(n), atol=1e-5)


def test_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        hadamard_matrix(3)
