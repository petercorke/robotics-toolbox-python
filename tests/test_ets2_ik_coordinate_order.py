"""Planar IK coordinates agree with the public forward-kinematics convention."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE2

from roboticstoolbox import ET2, ETS2

INDEX_CASES = [
    pytest.param((0, 1, 2), (0, 1, 2), id="sequential"),
    pytest.param((2, 0, 1), (1, 2, 0), id="cyclic-global"),
    pytest.param((1, 2, 0), (2, 0, 1), id="inverse-cyclic-global"),
    pytest.param((5, 2, 4), (0, 1, 2), id="sparse-compact"),
]


def arm(indices: tuple[int, ...], flipped: bool = False) -> ETS2:
    return ETS2(
        [
            ET2.R(jindex=indices[0], qlim=[0.1, 0.6]),
            ET2.tx(1),
            ET2.tx(jindex=indices[1], qlim=[6.5, 7.9]),
            ET2.R(jindex=indices[2], qlim=[-0.7, -0.1], flip=flipped),
            ET2.tx(0.7),
        ]
    )


def target(values: list[float], flipped: bool = False) -> SE2:
    first, extension, last = values
    return (
        SE2.Rot(first)
        * SE2.Tx(1 + extension)
        * SE2.Rot(-last if flipped else last)
        * SE2.Tx(0.7)
    )


@pytest.mark.parametrize("indices,order", INDEX_CASES)
@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("exact_start", [False, True])
def test_mixed_joint_ik_uses_consistent_coordinates(
    indices: tuple[int, ...], order: tuple[int, ...], flipped: bool, exact_start: bool
) -> None:
    ets = arm(indices, flipped)
    values = [0.35, 7.2, -0.4]
    desired = target(values, flipped)
    q = np.array(values)[list(order)]
    q0 = q if exact_start else np.array([0.2, 7.0, -0.3])[list(order)]
    sol = ets.ikine_LM(
        desired, q0=q0, slimit=1, ilimit=0 if exact_start else 80, tol=1e-12
    )

    assert sol.success, sol.reason
    nt.assert_allclose(sol.q, q, atol=5e-6)
    nt.assert_allclose(
        np.asarray(ets.fkine(sol.q).A, dtype=float), desired.A, atol=2e-6
    )
    nt.assert_array_equal(ets.jindices, indices)


@pytest.mark.parametrize("indices,order", INDEX_CASES)
def test_random_starts_obey_each_joints_limits(
    indices: tuple[int, ...], order: tuple[int, ...]
) -> None:
    ets = arm(indices)
    values = [0.35, 7.2, -0.4]
    sol = ets.ikine_LM(target(values), seed=3, slimit=8, ilimit=80, tol=1e-12)

    assert sol.success, sol.reason
    nt.assert_allclose(sol.q, np.array(values)[list(order)], atol=5e-6)


@pytest.mark.parametrize("indices,order", INDEX_CASES)
def test_trajectory_rows_use_the_same_coordinate_order(
    indices: tuple[int, ...], order: tuple[int, ...]
) -> None:
    ets = arm(indices)
    values = [[0.35, 7.2, -0.4], [0.42, 7.5, -0.5], [0.2, 6.8, -0.25]]
    desired = SE2([target(v) for v in values])
    q0 = np.array([0.2, 7.0, -0.3])[list(order)]
    sol = ets.ikine_LM(desired, q0=q0, slimit=1, ilimit=80, tol=1e-12)

    assert sol.success, sol.reason
    nt.assert_allclose(sol.q, np.array(values)[:, list(order)], atol=5e-6)


@pytest.mark.parametrize("indices,order", INDEX_CASES)
def test_limits_reject_the_same_physical_joint(
    indices: tuple[int, ...], order: tuple[int, ...]
) -> None:
    ets = arm(indices)
    values = [0.9, 7.2, -0.4]  # first revolute joint exceeds its upper bound
    q0 = np.array(values)[list(order)]
    rejected = ets.ikine_LM(target(values), q0=q0, slimit=1, ilimit=0, tol=1e-12)
    accepted = ets.ikine_LM(
        target(values), q0=q0, slimit=1, ilimit=0, tol=1e-12, joint_limits=False
    )

    assert not rejected.success
    assert accepted.success
    nt.assert_allclose(accepted.q[order.index(1)], values[1], atol=1e-12)
    nt.assert_allclose(
        np.asarray(ets.fkine(accepted.q).A, dtype=float), target(values).A, atol=1e-12
    )


def test_two_revolute_joints_with_swapped_indices_converge() -> None:
    ets = ET2.R(jindex=1) * ET2.tx(1) * ET2.R(jindex=0) * ET2.tx(0.8)
    desired = SE2.Rot(-0.4) * SE2.Tx(1) * SE2.Rot(0.7) * SE2.Tx(0.8)
    sol = ets.ikine_LM(desired, q0=[-0.2, 0.1], slimit=1, ilimit=80, tol=1e-12)

    assert sol.success, sol.reason
    nt.assert_allclose(sol.q, [0.7, -0.4], atol=5e-6)
