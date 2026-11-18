"""Independent gravity and velocity compensation in the task-space plant."""

from unittest.mock import Mock

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import models
from roboticstoolbox.blocks.arm import FDyn_X
from roboticstoolbox.robot import frne


@pytest.fixture(params=["python", "compiled"])
def dynamics_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.param == "compiled":
        if not frne._C_AVAILABLE:
            pytest.skip("compiled RNE extension is required")
        monkeypatch.setattr(
            models.DH.TwoLink,
            "rne_python",
            Mock(side_effect=AssertionError("compiled RNE fell back to Python")),
        )
    else:
        monkeypatch.setattr(frne, "_C_AVAILABLE", False)


@pytest.mark.parametrize(
    "gravcomp,velcomp",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["none", "gravity", "velocity", "both"],
)
@pytest.mark.parametrize("mdh", [False, True], ids=["standard", "modified"])
@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("loaded", [False, True])
def test_compensation_matches_closed_form(
    dynamics_backend: None,
    gravcomp: bool,
    velcomp: bool,
    mdh: bool,
    moving: bool,
    loaded: bool,
) -> None:
    robot = models.DH.TwoLink(mdh=mdh)
    q = np.array([0.3, 0.9])
    qd = np.array([0.4, -0.6]) if moving else np.zeros(2)
    wrench = np.array([0.4, -0.2, 0.1, 0.3, -0.1, 0.2]) if loaded else np.zeros(6)
    block = FDyn_X(robot, q, gravcomp=gravcomp, velcomp=velcomp)

    # Closed-form TwoLink equations in RTB's joint sign convention, as in
    # test_operational_space_closed_form.py: unit masses/lengths, half-link
    # centres of mass, zero link inertia, motor inertia and friction.
    c2 = np.cos(q[1])
    inertia = np.array([[1.5 + c2, 0.25 + 0.5 * c2], [0.25 + 0.5 * c2, 0.25]])
    h = -0.5 * np.sin(q[1])
    velocity_load = h * np.array([2 * qd[0] * qd[1] + qd[1] ** 2, -(qd[0] ** 2)])
    gravity_load = -9.8 * np.array(
        [1.5 * np.cos(q[0]) + 0.5 * np.cos(q.sum()), 0.5 * np.cos(q.sum())]
    )
    # The default base puts the arm in XZ; XYZ-RPY's varying angle is x[3].
    jacobian = np.zeros((6, 2))
    jacobian[0] = [-np.sin(q[0]) - np.sin(q.sum()), -np.sin(q.sum())]
    jacobian[2] = [np.cos(q[0]) + np.cos(q.sum()), np.cos(q.sum())]
    jacobian[3] = [1, 1]
    expected_force = jacobian.T @ wrench
    if not gravcomp:
        expected_force -= gravity_load
    if not velcomp:
        expected_force -= velocity_load
    expected_acceleration = np.linalg.solve(inertia, expected_force)

    derivative = block.test_deriv(wrench, x=np.r_[q, qd])

    nt.assert_array_equal(derivative[:2], qd)
    nt.assert_allclose(derivative[2:], expected_acceleration, atol=1e-11, rtol=1e-11)


def test_velocity_compensation_does_not_hold_a_stationary_puma_against_gravity() -> (
    None
):
    robot = models.DH.Puma560()
    q = np.array([0.3, -0.6, 0.7, 0.2, 0.8, -0.4])
    block = FDyn_X(robot, q, gravcomp=False, velcomp=True)
    expected = robot.accel(q, np.zeros(6), np.zeros(6))

    acceleration = block.test_deriv(np.zeros(6), x=np.r_[q, np.zeros(6)])[6:]

    assert np.linalg.norm(expected) > 1
    nt.assert_allclose(acceleration, expected, atol=1e-12)
