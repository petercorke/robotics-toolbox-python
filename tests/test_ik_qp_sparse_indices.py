"""QP iterations use only the active coordinates of a sparse ETS branch."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import ET, ETS, IK_QP
from roboticstoolbox.tools.types import NDArray
from tests import skip_no_qp


def _cartesian_chain(indices: list[int]) -> ETS:
    constructors = [ET.tx, ET.ty, ET.tz, ET.Rx, ET.Ry, ET.Rz]
    return ETS(
        [
            constructor(jindex=index, qlim=[-2.0, 2.0])
            for constructor, index in zip(constructors, indices)
        ]
    )


@skip_no_qp
@pytest.mark.parametrize("gain", [0.0, 1.0])
@pytest.mark.parametrize("indices", [[2, 4, 6, 8, 10, 12], [6, 1, 9, 3, 11, 4]])
def test_qp_step_matches_contiguous_chain(gain: float, indices: list[int]) -> None:
    contiguous = _cartesian_chain(list(range(6)))
    sparse = _cartesian_chain(indices)
    q_active: NDArray = np.array([1.9, -1.9, 0.1, 0.2, -0.1, 0.3])
    target = contiguous.fkine([1.8, -1.8, 0.2, 0.25, -0.08, 0.27]).A
    q_global = np.full(max(indices) + 1, 1.95)
    q_global[indices] = q_active
    inactive = np.ones(len(q_global), dtype=bool)
    inactive[indices] = False
    inactive_before = q_global[inactive].copy()
    solver = IK_QP(kq=gain, pi=0.3, ps=0.05)

    expected_error, expected_q = solver.step(contiguous, target, q_active.copy())
    actual_error, actual_q = solver.step(sparse, target, q_global)

    nt.assert_equal(actual_q.shape, (6,))
    nt.assert_allclose(actual_q, expected_q, atol=1e-10)
    nt.assert_allclose(actual_error, expected_error, atol=1e-12)
    nt.assert_array_equal(q_global[inactive], inactive_before)
    nt.assert_allclose(q_global[indices], actual_q, atol=1e-14)


@skip_no_qp
def test_qp_solves_sparse_chain_after_iterations() -> None:
    ets = _cartesian_chain([1, 3, 5, 7, 9, 11])
    target = ets.fkine([0.7, -0.2, 0.4, 0.2, -0.3, 0.4])
    result = ets.ikine_QP(
        target,
        q0=[0.5, -0.1, 0.3, 0.1, -0.2, 0.2],
        slimit=1,
        ilimit=100,
        tol=1e-12,
    )

    nt.assert_equal(result.success, True, err_msg=result.reason)
    nt.assert_array_less(0, result.iterations)
    nt.assert_equal(result.q.shape, (6,))
    nt.assert_allclose(ets.fkine(result.q).A, target.A, atol=2e-6)
