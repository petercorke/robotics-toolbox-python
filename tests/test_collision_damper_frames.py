"""Collision constraints agree with world-frame contact distance changes."""

import numpy as np
import pytest
from spatialgeometry import Cuboid, Sphere
from spatialmath import SE3

from roboticstoolbox import ET, Link, Robot, models
from tests import skip_no_collision_checking

pytestmark = skip_no_collision_checking


@pytest.mark.parametrize("base_angle", [0.0, np.pi / 2])
@pytest.mark.parametrize("end_angle", [0.0, np.pi / 2])
def test_prismatic_damper_is_invariant_to_frame_rotations(
    base_angle: float, end_angle: float
) -> None:
    root = Link(ET.tz(0), name="base")
    slider = Link(ET.tx(), name="slider", parent=root)
    end = Link(ET.Rz(end_angle), name="end", parent=slider, collision=[Sphere(0.05)])
    robot = Robot([root, slider, end], base=SE3.Rz(base_angle))
    q = np.array([0.1])
    normal = robot.base.R @ np.array([1.0, 0.0, 0.0])
    centre = robot.fkine(q, tool=SE3()).t
    obstacle = Sphere(0.05, pose=SE3(centre + 0.2 * normal))
    obstacle.v = [0.03, -0.04, 0.01, 0, 0, 0]
    distance, _, _ = robot.closest_point(q, obstacle)
    assert distance == pytest.approx(0.1)

    a, b = robot.link_collision_damper(obstacle, q, di=0.3, ds=0.05)

    # Positive slider velocity closes the gap at the same speed in every frame.
    np.testing.assert_allclose(a, [[1.0]], atol=1e-12)
    np.testing.assert_allclose(b, [0.2 + normal @ obstacle.v[:3]], atol=1e-12)


@pytest.mark.parametrize("rotated_base", [False, True])
@pytest.mark.parametrize("start_index", [None, 2])
def test_panda_damper_matches_contact_distance_gradient(
    rotated_base: bool, start_index: int | None
) -> None:
    robot = models.ETS.Panda()
    if rotated_base:
        robot.base = SE3(0.3, -0.2, 0.4) * SE3.RPY([0.3, -0.4, 0.7])
    # A robot-level end-effector tool must not offset a collision on link 4.
    robot.tool = SE3(0.2, 0.1, -0.15) * SE3.Rx(0.4)
    collision_pose = SE3(0.04, -0.03, 0.08) * SE3.Ry(0.6)
    robot.links[4].collision = [Sphere(0.05, pose=collision_pose)]
    q = robot.qr.copy()
    centre = robot.fkine(q, end=robot.links[4], tool=collision_pose).t
    obstacle = Sphere(0.05, pose=SE3(centre + [0.2, -0.1, 0.15]))
    expected = np.zeros(robot.n)
    step = 1e-6
    for j in range(0 if start_index is None else start_index, 5):
        dq = np.zeros(robot.n)
        dq[j] = step
        plus, _, _ = robot.closest_point(q + dq, obstacle)
        minus, _, _ = robot.closest_point(q - dq, obstacle)
        assert plus is not None and minus is not None
        expected[j] = -(plus - minus) / (2 * step)

    distance, _, _ = robot.closest_point(q, obstacle)
    assert distance is not None and 0.05 < distance < 0.5
    start = None if start_index is None else robot.links[start_index]
    a, b = robot.link_collision_damper(obstacle, q, start=start, di=0.5, ds=0.05)

    assert a.shape == (1, robot.n)
    np.testing.assert_allclose(a[0], expected, atol=1e-8)
    np.testing.assert_allclose(b, [(distance - 0.05) / 0.45], atol=1e-12)


def test_distant_collision_has_no_constraint() -> None:
    robot = models.ETS.Panda()
    robot.links[4].collision = [Sphere(0.05)]
    obstacle = Sphere(0.05, pose=SE3.Tx(10))
    robot.closest_point(robot.qr, obstacle)

    assert robot.link_collision_damper(obstacle, robot.qr) == (None, None)


def test_cuboid_damper_uses_contact_point_lever_arm() -> None:
    root = Link(ET.tz(0), name="base")
    collision_pose = SE3.Tx(0.8) * SE3.Rz(0.3)
    hinge = Link(
        ET.Rz(),
        parent=root,
        name="hinge",
        collision=[Cuboid([0.3, 0.6, 0.2], pose=collision_pose)],
    )
    robot = Robot([root, hinge])
    q = np.array([0.4])
    pose = robot.fkine(q, end=hinge, tool=collision_pose)
    # The nearest point lies inside the x face, displaced in y from its centre.
    obstacle = Sphere(0.05, pose=pose * SE3(0.5, 0.15, 0.02))
    step = 1e-6
    plus, _, _ = robot.closest_point(q + step, obstacle)
    minus, _, _ = robot.closest_point(q - step, obstacle)
    assert plus is not None and minus is not None
    expected = -(plus - minus) / (2 * step)
    # Using the cuboid centre instead would omit the -0.15 contact offset.
    assert expected == pytest.approx(0.8 * np.sin(0.3) - 0.15, abs=1e-8)
    distance, _, _ = robot.closest_point(q, obstacle)
    assert distance == pytest.approx(0.3)

    a, _ = robot.link_collision_damper(obstacle, q, di=0.5, ds=0.05)

    np.testing.assert_allclose(a, [[expected]], atol=1e-8)
