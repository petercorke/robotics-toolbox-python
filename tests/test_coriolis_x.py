"""Operational velocity terms must satisfy the transformed equations of motion."""

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.tools.types import NDArray


def _state(model: str) -> tuple[rtb.DHRobot, NDArray, NDArray, NDArray]:
    robot = getattr(rtb.models.DH, model)().nofriction(True, True)
    if model == "Panda":
        q = np.array([0.3, -0.5, 0.2, -1.8, 0.4, 1.5, 0.6])
    else:
        q = np.array([0.2, -0.7, 0.4, 0.3, 0.5, -0.2])
    qd = np.linspace(0.1, 0.4, robot.n)
    qdd = np.linspace(-0.3, 0.2, robot.n)
    return robot, q, qd, qdd


def _jacobian_rate(
    robot: rtb.DHRobot, q: NDArray, qd: NDArray, representation: str
) -> NDArray:
    # Independently differentiate along the actual joint velocity, rather than
    # using jacob0_dot, which is the implementation used by coriolis_x.
    step = 1e-5
    return (
        robot.jacob0_analytical(q + step * qd, representation=representation)
        - robot.jacob0_analytical(q - step * qd, representation=representation)
    ) / (2 * step)


@pytest.mark.parametrize("model", ["Puma560", "Panda"])
@pytest.mark.parametrize("representation", ["rpy/xyz", "rpy/zyx", "eul", "exp"])
def test_operational_coriolis_satisfies_equations_of_motion(
    model: str, representation: str
) -> None:
    robot, q, qd, qdd = _state(model)
    jacobian = robot.jacob0_analytical(q, representation=representation)
    inverse = np.linalg.pinv(jacobian)
    # For redundant robots use motions in the row space of J, so inverse
    # kinematics recovers the same qd/qdd without a null-space component.
    qd = inverse @ jacobian @ qd
    qdd = inverse @ jacobian @ qdd
    rate = _jacobian_rate(robot, q, qd, representation)
    velocity = jacobian @ qd
    acceleration = jacobian @ qdd + rate @ qd
    inertia = robot.inertia_x(q, representation=representation)
    wrench = inverse.T @ robot.rne(q, qd, qdd, gravity=[0, 0, 0])

    coriolis = robot.coriolis_x(q, qd, representation=representation)

    npt.assert_equal(coriolis.shape, (6, 6))
    npt.assert_allclose(
        inertia @ acceleration + coriolis @ velocity, wrench, atol=2e-6, rtol=2e-5
    )


def test_operational_coriolis_recomputes_each_trajectory_pose() -> None:
    robot, q0, qd0, _ = _state("Puma560")
    q = np.vstack((q0, q0 + np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1])))
    qd = np.vstack((qd0, -qd0))
    expected = np.stack([robot.coriolis_x(qk, qdk) for qk, qdk in zip(q, qd)])

    npt.assert_allclose(robot.coriolis_x(q, qd), expected, atol=1e-12)


@pytest.mark.parametrize("provided", ["J", "Ji", "Jd", "C", "Mx", "all"])
def test_operational_coriolis_accepts_precomputed_matrices(provided: str) -> None:
    robot, q, qd, _ = _state("Puma560")
    jacobian = robot.jacob0_analytical(q, representation="rpy/xyz")
    inverse = np.linalg.inv(jacobian)
    rate = _jacobian_rate(robot, q, qd, "rpy/xyz")
    joint_coriolis = robot.coriolis(q, qd)
    inertia = inverse.T @ robot.inertia(q) @ inverse
    matrices = {
        "J": jacobian,
        "Ji": inverse,
        "Jd": rate,
        "C": joint_coriolis,
        "Mx": inertia,
    }
    kwargs = matrices if provided == "all" else {provided: matrices[provided]}
    originals = {name: value.copy() for name, value in matrices.items()}
    expected = inverse.T @ joint_coriolis @ inverse - inertia @ rate @ inverse

    npt.assert_allclose(
        robot.coriolis_x(q, qd, **kwargs), expected, atol=2e-6, rtol=2e-5
    )
    for name, value in matrices.items():
        npt.assert_array_equal(value, originals[name])


@pytest.mark.parametrize("rows", [(2, 1), (1, 2)])
def test_operational_coriolis_rejects_mismatched_trajectory_rows(
    rows: tuple[int, int],
) -> None:
    robot, q, qd, _ = _state("Puma560")
    with pytest.raises(ValueError, match="same number of rows"):
        robot.coriolis_x(np.tile(q, (rows[0], 1)), np.tile(qd, (rows[1], 1)))
