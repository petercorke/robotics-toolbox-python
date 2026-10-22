"""Check operational-space dynamics against a closed-form planar 2R model.

Spong, Hutchinson and Vidyasagar, *Robot Modeling and Control* (Wiley,
2006), equation (7.87), supplies the joint-space equation
``tau = D @ qdd + C @ qd + g``. With ``xd = J @ qd``, differentiation gives
``xdd = J @ qdd + Jd @ qd``. For a square, nonsingular analytical Jacobian,
substitution of ``qdd = Ji @ (xdd - Jd @ qd)`` and ``F = Ji.T @ tau`` gives::

    Mx = Ji.T @ D @ Ji
    Cx = Ji.T @ C @ Ji - Mx @ Jd @ Ji
    gx = Ji.T @ g
    F = Mx @ xdd + Cx @ xd + gx

For TwoLink, J has full column rank and Ji is its Moore-Penrose inverse.
Since ``Ji @ J = I``, the force identity holds for every joint motion.
The 6-by-6 Mx is singular: forward acceleration is instead obtained from
``qdd = solve(D, J.T @ F - C @ qd - g)`` and the kinematic relation above.
Only the product ``Cx @ xd`` is unique, not the matrix factorisation of Cx.

TwoLink has unit link lengths and masses, centres of mass halfway along
each link, and zero friction and motor inertia. Its sign convention obeys
``tau_rtb(q, qd, qdd) = -tau_spong(-q, -qd, -qdd)``, as in the independent
RNE tests in test_fknm_fallback.py. Thus D and C have their usual planar
forms, while g has the opposite sign. Optional link inertias are those of
unit-mass cylinders of length 1 and radius 0.1 about their transverse axes.

The default base is Rx(pi/2), so the arm moves in the world XZ plane:
``p = (cos(q1) + cos(q1+q2), 0, sin(q1) + sin(q1+q2))``. For RTB's
``rpy/xyz`` ordering, ``R = Rx(pi/2) Rz(q1+q2)`` has coordinates
``(q1+q2, 0, pi/2)`` away from angle wrapping. Consequently J's nonzero
rows are 0, 2 and 3, with row 3 equal to (1, 1). Differentiating these
expressions by hand gives Jd. A geometric angular-velocity row would
instead lie along world -Y and is not interchangeable with this row.

All expected quantities below use these equations directly, without RTB's
inverse dynamics, Jacobians, inertia, Coriolis or gravity implementations.
"""

from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from roboticstoolbox.ets import fknm
from roboticstoolbox.models.DH import TwoLink
from roboticstoolbox.robot import frne
from roboticstoolbox.tools.types import NDArray

# Distinct rows exercise static, acceleration-only, velocity-only and combined
# loading. The straight-arm case still has full column rank when angle is included.
_STATES = np.array(
    [
        [[0.3, 0.5], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0.8, -0.4]],
        [[0.3, 0.9], [0.4, -0.6], [0, 0]],
        [[-1.1, 2.0], [-0.9, 1.3], [0.5, -0.7]],
        [[0.8, -1.2], [0.7, 0.2], [-0.3, 0.9]],
    ],
    dtype=np.float64,
)
_STATE_IDS = ["static", "straight-arm", "velocity", "mixed", "negative-elbow"]


@pytest.fixture(params=["python", "compiled"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise both backends, explicitly skipping unavailable extensions."""
    if request.param == "compiled":
        if not (fknm._C_AVAILABLE and frne._C_AVAILABLE):
            pytest.skip("compiled _fknm_c and _frne_c extensions are required")
        # RNE can silently fall back after an extension error. A compiled case
        # must fail in that situation rather than test the Python path twice.
        monkeypatch.setattr(
            TwoLink,
            "rne_python",
            Mock(side_effect=AssertionError("compiled RNE fell back to Python")),
        )
    else:
        monkeypatch.setattr(fknm, "_C_AVAILABLE", False)
        monkeypatch.setattr(frne, "_C_AVAILABLE", False)


@pytest.fixture(params=[0.0, (1 + 3 * 0.1**2) / 12], ids=["point-masses", "inertias"])
def link_inertia(request: pytest.FixtureRequest) -> float:
    """Return the independently specified transverse inertia of each link."""
    return request.param


@pytest.fixture(params=[False, True], ids=["standard-dh", "modified-dh"])
def robot(
    request: pytest.FixtureRequest, backend: None, link_inertia: float
) -> TwoLink:
    """Construct the same physical arm with either DH convention."""
    return TwoLink(mdh=request.param, inertia=link_inertia != 0)


def _joint_dynamics(
    q: NDArray, qd: NDArray, link_inertia: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Evaluate Spong's equations in TwoLink's joint sign convention.

    :param q: joint angles
    :param qd: joint velocities
    :param link_inertia: transverse inertia of each link about its centre of mass
    :returns: joint inertia D, Coriolis matrix C and gravity torque g
    """
    c2 = np.cos(q[1])
    inertia = np.array(
        [
            [1.5 + c2 + 2 * link_inertia, 0.25 + 0.5 * c2 + link_inertia],
            [0.25 + 0.5 * c2 + link_inertia, 0.25 + link_inertia],
        ]
    )
    h = -0.5 * np.sin(q[1])
    coriolis = h * np.array([[qd[1], qd[0] + qd[1]], [-qd[0], 0]])
    load = -9.8 * np.array(
        [1.5 * np.cos(q[0]) + 0.5 * np.cos(q.sum()), 0.5 * np.cos(q.sum())]
    )
    return inertia, coriolis, load


def _kinematics(q: NDArray, qd: NDArray) -> tuple[NDArray, NDArray]:
    """Differentiate the closed-form world position and XYZ RPY coordinates.

    :param q: joint angles
    :param qd: joint velocities
    :returns: analytical Jacobian J and its time derivative Jd
    """
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q.sum()), np.cos(q.sum())
    qd12 = qd.sum()
    jacobian = np.array(
        [
            [-s1 - s12, -s12],
            [0, 0],
            [c1 + c12, c12],
            [1, 1],
            [0, 0],
            [0, 0],
        ]
    )
    rate = np.zeros((6, 2))
    rate[0] = [-c1 * qd[0] - c12 * qd12, -c12 * qd12]
    rate[2] = [-s1 * qd[0] - s12 * qd12, -s12 * qd12]
    return jacobian, rate


@pytest.mark.parametrize("state", _STATES, ids=_STATE_IDS)
def test_inertia_matches_closed_form(
    robot: TwoLink, link_inertia: float, state: NDArray
) -> None:
    q, qd, _ = state
    inertia, _, _ = _joint_dynamics(q, qd, link_inertia)
    jacobian, _ = _kinematics(q, qd)
    inverse = np.linalg.pinv(jacobian)
    expected = inverse.T @ inertia @ inverse

    result = robot.inertia_x(q)

    npt.assert_equal(result.shape, (6, 6))
    npt.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("state", _STATES, ids=_STATE_IDS)
def test_operational_forces_match_spong(
    robot: TwoLink, link_inertia: float, state: NDArray
) -> None:
    q, qd, qdd = state
    inertia, coriolis, gravity = _joint_dynamics(q, qd, link_inertia)
    jacobian, rate = _kinematics(q, qd)
    inverse = np.linalg.pinv(jacobian)
    velocity = jacobian @ qd
    acceleration = jacobian @ qdd + rate @ qd
    expected_inertia = inverse.T @ inertia @ inverse
    expected_velocity_force = inverse.T @ (coriolis @ qd) - expected_inertia @ rate @ qd
    expected_wrench = inverse.T @ (inertia @ qdd + coriolis @ qd + gravity)

    mx = robot.inertia_x(q)
    cx = robot.coriolis_x(q, qd)
    gx = robot.gravload_x(q)

    npt.assert_equal(cx.shape, (6, 6))
    npt.assert_equal(gx.shape, (6,))
    # Check each force separately so errors cannot cancel in the total wrench.
    # Terms involving RTB's numerical Jacobian derivative allow roundoff from
    # finite differences; inertia and gravity use tighter tolerances.
    npt.assert_allclose(cx @ velocity, expected_velocity_force, atol=2e-8, rtol=2e-7)
    npt.assert_allclose(gx, inverse.T @ gravity, atol=1e-12, rtol=1e-12)
    npt.assert_allclose(
        mx @ acceleration + cx @ velocity + gx,
        expected_wrench,
        atol=2e-8,
        rtol=2e-7,
    )


@pytest.mark.parametrize("state", _STATES, ids=_STATE_IDS)
def test_acceleration_recovers_closed_form_joint_motion(
    robot: TwoLink, link_inertia: float, state: NDArray
) -> None:
    q, qd, qdd = state
    inertia, coriolis, gravity = _joint_dynamics(q, qd, link_inertia)
    jacobian, rate = _kinematics(q, qd)
    wrench = np.linalg.pinv(jacobian).T @ (inertia @ qdd + coriolis @ qd + gravity)
    expected = jacobian @ qdd + rate @ qd

    result = robot.accel_x(q, jacobian @ qd, wrench)

    npt.assert_equal(result.shape, (6,))
    npt.assert_allclose(result, expected, atol=2e-8, rtol=2e-7)


def test_trajectory_matches_closed_form_at_each_state(
    robot: TwoLink, link_inertia: float
) -> None:
    q, qd, qdd = _STATES.transpose(1, 0, 2)
    velocities: list[NDArray] = []
    accelerations: list[NDArray] = []
    wrenches: list[NDArray] = []
    inertias: list[NDArray] = []
    velocity_forces: list[NDArray] = []
    gravity_forces: list[NDArray] = []
    for qk, qdk, qddk in zip(q, qd, qdd):
        inertia, coriolis, gravity = _joint_dynamics(qk, qdk, link_inertia)
        jacobian, rate = _kinematics(qk, qdk)
        inverse = np.linalg.pinv(jacobian)
        mx = inverse.T @ inertia @ inverse
        inertias.append(mx)
        velocity_forces.append(inverse.T @ (coriolis @ qdk) - mx @ rate @ qdk)
        gravity_forces.append(inverse.T @ gravity)
        velocities.append(jacobian @ qdk)
        accelerations.append(jacobian @ qddk + rate @ qdk)
        wrenches.append(inverse.T @ (inertia @ qddk + coriolis @ qdk + gravity))
    velocity = np.array(velocities)
    acceleration = np.array(accelerations)
    wrench = np.array(wrenches)

    mx = robot.inertia_x(q)
    cx = robot.coriolis_x(q, qd)
    gx = robot.gravload_x(q)
    xdd = robot.accel_x(q, velocity, wrench)

    npt.assert_equal(mx.shape, (len(q), 6, 6))
    npt.assert_equal(cx.shape, (len(q), 6, 6))
    npt.assert_equal(gx.shape, (len(q), 6))
    npt.assert_equal(xdd.shape, (len(q), 6))
    npt.assert_allclose(mx, inertias, atol=1e-12, rtol=1e-12)
    npt.assert_allclose(gx, gravity_forces, atol=1e-12, rtol=1e-12)
    cx_velocity = (cx @ velocity[..., None])[..., 0]
    npt.assert_allclose(cx_velocity, velocity_forces, atol=2e-8, rtol=2e-7)
    npt.assert_allclose(
        (mx @ acceleration[..., None])[..., 0] + cx_velocity + gx,
        wrench,
        atol=2e-8,
        rtol=2e-7,
    )
    npt.assert_allclose(xdd, acceleration, atol=2e-8, rtol=2e-7)
