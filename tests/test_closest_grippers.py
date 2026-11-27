"""Robot distance queries include independently configured gripper geometry."""

import numpy as np
import pytest
from spatialgeometry import Sphere
from spatialmath import SE3

from roboticstoolbox import ET, Link, Robot
from tests import skip_no_collision_checking

pytestmark = skip_no_collision_checking


def gripper_robot(arm_geometry: bool = True) -> Robot:
    arm = Link(ET.Rz(), name="arm", collision=[Sphere(0.1)] if arm_geometry else [])
    finger = Link(
        ET.tx(1) * ET.tx(), parent=arm, name="finger", collision=[Sphere(0.1)]
    )
    robot = Robot([arm, finger], gripper_links=finger)
    robot.grippers[0].q = [0.2]
    return robot


@pytest.mark.parametrize("arm_geometry", [False, True])
@pytest.mark.parametrize("obstacle_x, expected", [(1.6, 0.2), (1.3, -0.1)])
def test_gripper_is_the_closest_collision_body(
    arm_geometry: bool, obstacle_x: float, expected: float
) -> None:
    robot = gripper_robot(arm_geometry)
    obstacle = Sphere(0.1, pose=SE3.Tx(obstacle_x))

    distance, robot_point, obstacle_point = robot.closest_point(
        np.zeros(1), obstacle, inf_dist=0.5
    )

    assert distance == pytest.approx(expected)
    np.testing.assert_allclose(robot_point, [1.3, 0, 0], atol=1e-9)
    np.testing.assert_allclose(obstacle_point, [obstacle_x - 0.1, 0, 0], atol=1e-9)
    assert robot.iscollided(np.zeros(1), obstacle) == (expected < 0)


def test_arm_can_remain_closer_than_the_gripper() -> None:
    robot = gripper_robot()
    obstacle = Sphere(0.1, pose=SE3.Tx(-0.4))

    distance, robot_point, obstacle_point = robot.closest_point(
        np.zeros(1), obstacle, inf_dist=2
    )

    assert distance == pytest.approx(0.2)
    np.testing.assert_allclose(robot_point, [-0.1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(obstacle_point, [-0.3, 0, 0], atol=1e-9)


def test_distance_follows_each_grippers_independent_configuration() -> None:
    arm = Link(ET.Rz(), name="arm")
    first = Link(ET.tx(), parent=arm, name="first", collision=[Sphere(0.1)])
    second = Link(ET.tx(), parent=arm, name="second", collision=[Sphere(0.1)])
    robot = Robot([arm, first, second], gripper_links=[first, second])
    robot.base = SE3(0.4, -0.2, 0.3) * SE3.Rz(np.pi / 2)
    robot.grippers[0].q = [-1.0]
    robot.grippers[1].q = [1.2]
    obstacle = Sphere(0.1, pose=robot.base * SE3.Tx(1.6))

    distance, robot_point, obstacle_point = robot.closest_point(np.zeros(1), obstacle)

    assert distance == pytest.approx(0.2)
    np.testing.assert_allclose(robot_point, (robot.base * SE3.Tx(1.3)).t, atol=1e-9)
    np.testing.assert_allclose(obstacle_point, (robot.base * SE3.Tx(1.5)).t, atol=1e-9)
    robot.grippers[1].q = [-1.0]
    assert robot.closest_point(np.zeros(1), obstacle) == (None, None, None)
    robot.grippers[0].q = [1.4]
    assert robot.closest_point(np.zeros(1), obstacle)[0] == pytest.approx(0.0)


def test_skip_uses_the_grippers_existing_world_transforms() -> None:
    robot = gripper_robot()
    obstacle = Sphere(0.1, pose=SE3.Tx(1.6))
    robot.closest_point(np.zeros(1), obstacle)
    robot.grippers[0].q = [0.3]

    assert robot.closest_point(np.array([np.pi]), obstacle, skip=True)[
        0
    ] == pytest.approx(0.2)
    assert robot.closest_point(np.zeros(1), obstacle)[0] == pytest.approx(0.1)


def test_empty_and_distant_gripper_geometry_give_no_result() -> None:
    robot = gripper_robot(arm_geometry=False)
    obstacle = Sphere(0.1, pose=SE3.Tx(10))
    assert robot.closest_point(np.zeros(1), obstacle) == (None, None, None)
    robot.grippers[0].links[0].collision = []
    obstacle.T = SE3.Tx(1.3)
    assert robot.closest_point(np.zeros(1), obstacle) == (None, None, None)
