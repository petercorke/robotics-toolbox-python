"""Prismatic joint offsets must contribute to inverse-dynamics moment arms."""

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.robot.frne import _C_AVAILABLE


def _point_mass_arm(mdh: bool, offset: float) -> rtb.DHRobot:
    if mdh:
        links = [
            rtb.RevoluteMDH(),
            rtb.PrismaticMDH(a=1, alpha=np.pi / 2, offset=offset, m=2),
        ]
    else:
        links = [
            rtb.RevoluteDH(a=1, alpha=np.pi / 2),
            rtb.PrismaticDH(offset=offset, m=2),
        ]
    return rtb.DHRobot(links, gravity=[0, 0, 0])


@pytest.mark.parametrize("mdh", [False, True])
@pytest.mark.parametrize("offset", [-0.4, 0.0, 0.4])
def test_prismatic_offset_matches_point_mass_inertia(mdh: bool, offset: float) -> None:
    robot = _point_mass_arm(mdh, offset)
    q = np.array([0.3, 0.3])
    qdd = np.array([0.7, -0.2])
    d = q[1] + offset
    # The point mass is at Rz(q1) @ [1, -d, 0], so M = m * J.T @ J.
    inertia = 2 * np.array([[1 + d**2, -1], [-1, 1]])

    npt.assert_allclose(
        robot.rne_python(q, np.zeros(2), qdd), inertia @ qdd, atol=1e-13
    )


@pytest.mark.parametrize("mdh", [False, True])
def test_prismatic_offset_equals_shifted_coordinate(mdh: bool) -> None:
    offset = 0.4
    robot = _point_mass_arm(mdh, offset)
    reference = _point_mass_arm(mdh, 0)
    q = np.array([[0.3, 0.2], [-0.5, 0.6]])
    qd = np.array([[0.4, -0.3], [-0.1, 0.2]])
    qdd = np.array([[0.7, -0.2], [0.3, 0.5]])
    shifted = q + np.array([0, offset])
    gravity = np.array([1.2, -3.4, -7.8])
    wrench = np.array([0.2, -0.3, 0.4, -0.1, 0.5, 0.7])

    npt.assert_allclose(robot.fkine(q).A, reference.fkine(shifted).A)
    npt.assert_allclose(
        robot.rne_python(q, qd, qdd, gravity=gravity, fext=wrench),
        reference.rne_python(shifted, qd, qdd, gravity=gravity, fext=wrench),
        atol=1e-13,
    )


@pytest.mark.skipif(not _C_AVAILABLE, reason="compiled _frne_c extension not built")
@pytest.mark.parametrize("mdh", [False, True])
def test_prismatic_offset_python_matches_compiled_path(mdh: bool) -> None:
    robot = _point_mass_arm(mdh, 0.4)
    q = np.array([[0.3, 0.2], [-0.5, 0.6]])
    qd = np.array([[0.4, -0.3], [-0.1, 0.2]])
    qdd = np.array([[0.7, -0.2], [0.3, 0.5]])

    compiled = robot.rne(q, qd, qdd, base_wrench=True)
    python = robot.rne_python(q, qd, qdd, base_wrench=True)

    for actual, expected in zip(python, compiled):
        npt.assert_allclose(actual, expected, atol=1e-13)


@pytest.mark.skipif(not _C_AVAILABLE, reason="compiled _frne_c extension not built")
@pytest.mark.parametrize("mdh", [False, True])
def test_zero_offset_flipped_joint_preserves_existing_native_parity(mdh: bool) -> None:
    # This offset fix must preserve existing zero-offset behaviour. General
    # flipped-joint dynamics also require changes to velocity/effort signs and
    # the compiled link representation, which are outside this regression.
    if mdh:
        links = [
            rtb.RevoluteMDH(),
            rtb.PrismaticMDH(alpha=-np.pi / 2, flip=True, m=1.7, r=[0, 0, 0.2]),
        ]
    else:
        links = [
            rtb.RevoluteDH(alpha=-np.pi / 2),
            rtb.PrismaticDH(flip=True, m=1.7, r=[0, 0, 0.2]),
        ]
    robot = rtb.DHRobot(links, gravity=[0, 0, 0])
    q = [0.3, 0.6]
    qd = [0.4, 0.3]
    qdd = [0.7, -0.2]

    native = robot.rne(q, qd, qdd)

    npt.assert_allclose(native, [1.088, -0.5576], atol=1e-13)
    npt.assert_allclose(robot.rne_python(q, qd, qdd), native, atol=1e-13)
