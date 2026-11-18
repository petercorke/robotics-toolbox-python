"""The rendered ellipsoids describe world-space velocity and force capability."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath import SE3

import roboticstoolbox as rtb


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def numerical_world_jacobian(robot, q):
    """Differentiate forward kinematics without using either Jacobian method."""
    pose = robot.fkine(q)
    step = 1e-6
    jacobian = np.empty((6, robot.n))
    for joint in range(robot.n):
        delta = np.zeros(robot.n)
        delta[joint] = step
        plus = robot.fkine(q + delta)
        minus = robot.fkine(q - delta)
        jacobian[:3, joint] = (plus.t - minus.t) / (2 * step)
        angular = ((plus.R - minus.R) / (2 * step)) @ pose.R.T
        jacobian[3:, joint] = angular[[2, 0, 1], [1, 2, 0]]
    return jacobian


def relative_surface(ellipse, centre):
    points = np.vstack([ellipse.x.ravel(), ellipse.y.ravel(), ellipse.z.ravel()])
    return (points - np.asarray(centre)[:, None]) / ellipse.scale


def assert_unit_capability(robot, q, ellipse, centre):
    jacobian = numerical_world_jacobian(robot, q)
    jacobian = jacobian[:3] if ellipse.opt == "trans" else jacobian[3:]
    points = relative_surface(ellipse, centre)
    if ellipse.vell:
        # The minimum-norm joint velocity to attain each surface velocity has
        # unit norm.  This is independent of the renderer's SVD construction.
        joint_rates = np.linalg.lstsq(jacobian, points, rcond=None)[0]
        assert_allclose(jacobian @ joint_rates, points, atol=2e-8)
        assert_allclose(np.linalg.norm(joint_rates, axis=0), 1, atol=2e-7)
    else:
        # Virtual work maps each surface wrench to unit-norm joint effort.
        assert_allclose(np.linalg.norm(jacobian.T @ points, axis=0), 1, atol=2e-7)


@pytest.mark.parametrize(
    "model", [rtb.models.DH.Puma560, rtb.models.DH.Panda, rtb.models.Panda]
)
@pytest.mark.parametrize("etype", ["v", "f"])
@pytest.mark.parametrize("opt", ["trans", "rot"])
@pytest.mark.parametrize("placed", [False, True])
def test_rendered_model_ellipsoids_match_fk_velocity_and_virtual_work(
    model, etype, opt, placed
):
    robot = model()
    q = np.array(robot.qn if hasattr(robot, "qn") else robot.qr)
    q = q + np.linspace(0.03, 0.12, robot.n)
    if placed:
        robot.base = SE3(0.4, -0.3, 0.2) * SE3.RPY([0.4, -0.2, 0.7])
        robot.tool = robot.tool * SE3(0.1, -0.05, 0.2) * SE3.RPY([0.2, 0.3, -0.4])

    create = robot.vellipse if etype == "v" else robot.fellipse
    for centre in ("ee", [1.2, -0.3, 0.4]):
        ellipse = create(q, opt=opt, centre=centre, add=False)
        ellipse.scale = 0.4
        ax = plt.figure().add_subplot(projection="3d")
        ellipse.plot(ax=ax)
        ax.figure.canvas.draw()
        assert ellipse.ell is not None
        position = robot.fkine(q).t if isinstance(centre, str) else centre
        assert_unit_capability(robot, q, ellipse, position)


@pytest.mark.parametrize("q", [np.array([0.0, 0.0]), np.array([-0.5, 0.6])])
@pytest.mark.parametrize("placed", [False, True])
def test_rank_deficient_velocity_ellipsoid_stays_in_attainable_subspace(q, placed):
    robot = rtb.DHRobot([rtb.RevoluteDH(a=1), rtb.RevoluteDH(a=0.7)])
    if placed:
        robot.base = SE3(0.2, 0.3, -0.1) * SE3.RPY([0.5, -0.3, 0.2])
    ellipse = robot.vellipse(q, scale=1, add=False)
    assert ellipse.make_ellipsoid()
    points = relative_surface(ellipse, robot.fkine(q).t)
    jacobian = numerical_world_jacobian(robot, q)[:3]
    joint_rates = np.linalg.lstsq(jacobian, points, rcond=1e-8)[0]
    assert_allclose(jacobian @ joint_rates, points, atol=2e-8)
    speeds = np.linalg.norm(joint_rates, axis=0)
    assert speeds.max() <= 1 + 2e-8
    assert speeds.max() > 0.999


@pytest.mark.parametrize("q", [[0.0, 0.0], [-0.5, 0.6]])
def test_unbounded_force_ellipsoid_is_not_drawn(q):
    robot = rtb.DHRobot([rtb.RevoluteDH(a=1), rtb.RevoluteDH(a=0.7)])
    ellipse = robot.fellipse(q, add=False)
    with pytest.warns(RuntimeWarning, match="Force ellipsoid is singular"):
        assert not ellipse.make_ellipsoid()


def test_zero_angular_capability_is_a_single_point():
    robot = rtb.DHRobot([rtb.PrismaticDH()])
    ellipse = robot.vellipse([0.4], opt="rot", scale=1, add=False)
    assert ellipse.make_ellipsoid()
    assert_allclose(relative_surface(ellipse, robot.fkine([0.4]).t), 0, atol=1e-14)


@pytest.mark.parametrize("etype", ["v", "f"])
def test_redraw_updates_the_capability_and_replaces_the_artist(etype):
    robot = rtb.models.DH.Puma560()
    create = robot.vellipse if etype == "v" else robot.fellipse
    ellipse = create(robot.qn, add=False)
    ax = plt.figure().add_subplot(projection="3d")
    ellipse.plot(ax=ax)
    original = ellipse.ell
    q = robot.qn + np.array([0.1, -0.2, 0.15, 0.2, 0.1, -0.1])
    ellipse.q = q
    ellipse.draw()
    ax.figure.canvas.draw()
    assert ellipse.ell is not original
    assert len(ax.collections) == 1
    assert_unit_capability(robot, q, ellipse, robot.fkine(q).t)
