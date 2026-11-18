"""DH joint twists preserve the same world-space motion as forward kinematics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath import SE3

import roboticstoolbox as rtb


def mixed_robot(mdh, convention):
    revolute = rtb.RevoluteMDH if mdh else rtb.RevoluteDH
    prismatic = rtb.PrismaticMDH if mdh else rtb.PrismaticDH
    flip = convention in ("flip", "all")
    robot = rtb.DHRobot(
        [
            revolute(a=0.4, alpha=0.3, d=0.2, offset=0.1, flip=flip),
            prismatic(a=0.2, alpha=-0.4, theta=0.5, offset=0.15, flip=flip),
            revolute(a=0.3, alpha=0.2, d=-0.1, offset=-0.2, flip=flip),
        ]
    )
    if convention in ("base", "all"):
        robot.base = SE3(0.7, -0.3, 0.4) * SE3.RPY([0.4, -0.2, 0.3])
    if convention in ("tool", "all"):
        robot.tool = SE3(0.2, 0.1, -0.15) * SE3.RPY([-0.2, 0.3, 0.4])
    return robot


def product_of_exponentials(twists, reference, displacement):
    pose = SE3()
    for twist, delta in zip(twists, displacement):
        pose = pose * twist.exp(delta)
    return pose * reference


@pytest.mark.parametrize("mdh", [False, True])
@pytest.mark.parametrize("convention", ["plain", "base", "tool", "flip", "all"])
@pytest.mark.parametrize("qref", [np.zeros(3), np.array([0.2, -0.1, 0.3])])
def test_twists_reconstruct_displaced_dh_motion(mdh, convention, qref):
    robot = mixed_robot(mdh, convention)
    twists, reference = robot.twists(qref)

    assert len(twists) == robot.n
    assert_allclose(reference.A, robot.fkine(qref).A, atol=2e-14)
    for displacement in ([0.17, 0.05, -0.12], [-0.3, -0.2, 0.4]):
        reconstructed = product_of_exponentials(twists, reference, displacement)
        assert_allclose(reconstructed.A, robot.fkine(qref + displacement).A, atol=2e-14)


@pytest.mark.parametrize("mdh", [False, True])
@pytest.mark.parametrize("convention", ["plain", "base", "tool", "flip", "all"])
def test_twists_match_world_velocity_from_finite_differences(mdh, convention):
    robot = mixed_robot(mdh, convention)
    qref = np.array([0.2, -0.1, 0.3])
    twists, _ = robot.twists(qref)
    pose = robot.fkine(qref)
    step = 1e-6
    for joint, twist in enumerate(twists):
        delta = np.zeros(robot.n)
        delta[joint] = step
        plus = robot.fkine(qref + delta)
        minus = robot.fkine(qref - delta)
        linear = (plus.t - minus.t) / (2 * step)
        angular_matrix = ((plus.R - minus.R) / (2 * step)) @ pose.R.T
        angular = angular_matrix[[2, 0, 1], [1, 2, 0]]
        assert_allclose(twist.v + np.cross(twist.w, pose.t), linear, atol=2e-9)
        assert_allclose(twist.w, angular, atol=2e-9)


@pytest.mark.parametrize("model", [rtb.models.DH.Puma560, rtb.models.DH.Panda])
@pytest.mark.parametrize("placed", [False, True])
def test_model_twists_construct_equivalent_poe_robot(model, placed):
    robot = model()
    if placed:
        robot.base = SE3(-0.3, 0.5, 0.1) * SE3.RPY([0.2, -0.3, 0.4])
    twists, reference = robot.twists()
    links = [rtb.PoERevolute(twist.w, np.cross(twist.w, twist.v)) for twist in twists]
    poe = rtb.PoERobot(links, reference)
    for configuration in (robot.qz, robot.qr, np.linspace(-0.3, 0.4, robot.n)):
        assert_allclose(
            poe.fkine(configuration).A, robot.fkine(configuration).A, atol=2e-14
        )


@pytest.mark.parametrize("mdh", [False, True])
def test_first_prismatic_joint_and_default_reference(mdh):
    prismatic = rtb.PrismaticMDH if mdh else rtb.PrismaticDH
    robot = rtb.DHRobot(
        [prismatic(a=0.2, alpha=0.4, offset=0.3, flip=True)],
        base=SE3(0.4, -0.2, 0.3) * SE3.Rx(0.5),
        tool=SE3.Tx(0.2),
    )
    robot.q = [0.8]
    original_base = robot.base.A.copy()
    original_tool = robot.tool.A.copy()
    twists, reference = robot.twists()
    assert_allclose(reference.A, robot.fkine([0]).A, atol=2e-14)
    reconstructed = product_of_exponentials(twists, reference, [0.35])
    assert_allclose(reconstructed.A, robot.fkine([0.35]).A, atol=2e-14)
    assert_allclose(robot.q, [0.8])
    assert_allclose(robot.base.A, original_base)
    assert_allclose(robot.tool.A, original_tool)
