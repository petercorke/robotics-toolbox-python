"""A half-turn must preserve its arbitrary axis in the pose error."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox.ets import fknm

ERROR_FUNCTIONS = [
    rtb.angle_axis,
    rtb.angle_axis_python,
    pytest.param(
        fknm.Angle_Axis,
        marks=pytest.mark.skipif(
            not fknm._C_AVAILABLE, reason="requires the compiled kinematics extension"
        ),
    ),
]


@pytest.mark.parametrize("axis", [(1, 2, -2), (-2, 1, 3), (1, 1, 0), (0, 0, 1)])
@pytest.mark.parametrize("error_function", ERROR_FUNCTIONS)
def test_half_turn_error_reconstructs_the_desired_orientation(axis, error_function):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    orientation = SE3.RPY(0.3, -0.5, 0.7)
    current = SE3(0.2, -0.4, 0.8) * orientation
    desired = SE3(-0.3, 0.1, 0.5) * SE3.AngVec(np.pi, axis) * orientation

    error = error_function(current.A, desired.A)

    nt.assert_allclose(error[:3], [-0.5, 0.5, -0.3], atol=1e-14)
    nt.assert_allclose(np.linalg.norm(error[3:]), np.pi, atol=1e-14)
    recovered = SE3.AngVec(np.linalg.norm(error[3:]), error[3:])
    nt.assert_allclose(recovered.R @ current.R, desired.R, atol=1e-14)


@pytest.mark.parametrize("as_matrix", [False, True])
def test_p_servo_half_turn_uses_the_same_rotation_error(as_matrix):
    axis = np.array([1.0, -2.0, 2.0]) / 3
    target = SE3.AngVec(np.pi, axis)
    start = SE3()
    if as_matrix:
        start, target = start.A, target.A

    velocity, arrived = rtb.p_servo(
        start, target, gain=2, threshold=0.1, method="angle-axis"
    )

    nt.assert_allclose(np.linalg.norm(velocity[3:]), 2 * np.pi, atol=1e-14)
    nt.assert_equal(arrived, False)
    recovered = SE3.AngVec(np.linalg.norm(velocity[3:] / 2), velocity[3:])
    nt.assert_allclose(recovered.R, SE3.AngVec(np.pi, axis).R, atol=1e-14)


@pytest.mark.parametrize(
    "solver_name", ["ik_LM", "ik_GN", "ik_NR", "ikine_LM", "ikine_GN", "ikine_NR"]
)
def test_ik_does_not_accept_an_underestimated_half_turn_residual(solver_name):
    if solver_name.startswith("ik_") and not fknm._C_AVAILABLE:
        pytest.skip("requires the compiled kinematics extension")
    chain = rtb.ET.Rx() * rtb.ET.Ry() * rtb.ET.Rz()
    target = SE3.AngVec(np.pi, np.array([1.0, -2.0, 2.0]) / 3)

    result = getattr(chain, solver_name)(
        target,
        q0=np.zeros(3),
        mask=np.array([0, 0, 0, 1, 1, 1]),
        ilimit=1,
        slimit=1,
        tol=3.0,
        joint_limits=False,
    )

    nt.assert_equal(result.success, False)
    nt.assert_array_less(3.0, result.residual)
    if solver_name.startswith("ik_"):
        nt.assert_allclose(result.residual, np.pi**2 / 2, atol=1e-14)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.7, np.pi - 1e-7, np.pi + 1e-7])
@pytest.mark.parametrize("error_function", ERROR_FUNCTIONS)
def test_rotation_error_is_continuous_away_from_the_half_turn(angle, error_function):
    target = SE3.AngVec(angle, np.array([1.0, -2.0, 2.0]) / 3)
    error = error_function(SE3().A, target.A)

    magnitude = np.linalg.norm(error[3:])
    nt.assert_allclose(magnitude, min(angle, 2 * np.pi - angle), atol=1e-14)
    recovered = SE3.AngVec(magnitude, error[3:])
    nt.assert_allclose(recovered.R, target.R, atol=1e-14)
