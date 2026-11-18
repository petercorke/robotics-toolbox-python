"""Zero constraints must leave the full configuration space available."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import null


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (0, 4), (4, 0), (0, 0)])
@pytest.mark.parametrize("rtol", [0, 1e-12])
def test_zero_rank_nullspace(shape: tuple[int, int], rtol: float) -> None:
    matrix = np.zeros(shape)
    basis = null(matrix, atol=0, rtol=rtol)
    assert basis.shape == (shape[1], shape[1])
    nt.assert_allclose(matrix @ basis, np.zeros(shape))
    nt.assert_allclose(basis.T @ basis, np.eye(shape[1]))


def test_exact_zero_singular_values_do_not_count_towards_rank() -> None:
    matrix = np.diag([2.0, 0, 0])
    basis = null(matrix, atol=0, rtol=0)
    assert basis.shape == (3, 2)
    nt.assert_allclose(matrix @ basis, np.zeros((3, 2)))
    nt.assert_allclose(basis.T @ basis, np.eye(2))


def test_nonzero_singular_value_at_the_tolerance_is_retained() -> None:
    matrix = np.diag([2.0, 1.0, 0.0])
    basis = null(matrix, atol=1, rtol=0)
    assert basis.shape == (3, 1)
    nt.assert_allclose(matrix @ basis, np.zeros((3, 1)))
