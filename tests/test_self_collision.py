"""
Tests for robot self-collision checking.

Topology tests check pair selection independently of the collision engine.
Geometry tests use Coal with simple shapes whose intersections are known
analytically, including nested groups, grippers and repeated pose updates.
"""

from unittest.mock import patch

import numpy as np
import pytest
from spatialgeometry import CollisionShapeGroup, Cuboid, SceneGroup, Sphere
from spatialmath import SE3

from roboticstoolbox import ET, DHRobot, Link, RevoluteDH, Robot, models
from roboticstoolbox.tools.types import ArrayLike
from tests import skip_no_collision_checking


def chain() -> Robot:
    """
    Construct a two-joint chain with collision geometry on all three links

    :returns: robot with two unit-length segments and spherical geometry
    """
    # The tip centre moves from x=2 to the base centre as q goes from 0 to pi.
    base = Link(ET.Rz(), name="base", collision=[Sphere(0.2)])
    # A distant middle body keeps base and tip non-adjacent without colliding.
    middle = Link(
        ET.tx(1) * ET.Rz(),
        parent=base,
        name="middle",
        collision=[Sphere(0.1, pose=SE3.Tz(10))],
    )
    tip = Link(ET.tx(1), parent=middle, name="tip", collision=[Sphere(0.2)])
    return Robot([base, middle, tip])


def test_model_without_collision_geometry_raises() -> None:
    robot = Robot([Link(ET.Rz(), geometry=[Sphere(1)])], name="visual only")
    with pytest.raises(ValueError, match="no collision geometry"):
        robot.isselfcollided([0])
    with pytest.raises(ValueError, match="no collision geometry"):
        robot.iscollided([0], robot)


def test_reported_dh_panda_example_raises_clear_error() -> None:
    robot = models.DH.Panda()
    with pytest.raises(ValueError, match="no collision geometry"):
        robot.iscollided(robot.q, robot)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("group_type", [CollisionShapeGroup, SceneGroup])
def test_empty_nested_groups_are_not_collision_geometry(
    depth: int, group_type: type[CollisionShapeGroup] | type[SceneGroup]
) -> None:
    group = group_type()
    for _ in range(depth):
        group = group_type([group])
    robot = Robot([Link(ET.Rz(), collision=group)])
    with pytest.raises(ValueError, match="no collision geometry"):
        robot.isselfcollided([0])


def test_another_robot_is_not_accepted_as_an_external_shape() -> None:
    robot = chain()
    with pytest.raises(TypeError, match="external Shape or this robot"):
        robot.iscollided([0, 0], chain())


def test_collision_geometry_added_after_construction_is_used() -> None:
    robot = Robot([Link(ET.Rz())])
    robot.links[0].collision.append(Sphere(1))
    assert robot.isselfcollided([0]) is False


def test_geometry_on_one_link_cannot_collide_with_itself() -> None:
    robot = Robot([Link(ET.Rz(), collision=[Sphere(1), Sphere(1)])])
    with patch.object(Sphere, "iscollided") as collided:
        assert robot.isselfcollided([0]) is False
    collided.assert_not_called()


def test_adjacent_links_are_not_checked() -> None:
    base = Link(ET.Rz(), collision=[Sphere(1)])
    tip = Link(ET.Rz(), parent=base, collision=[Sphere(1)])
    robot = Robot([base, tip])
    with patch.object(Sphere, "iscollided") as collided:
        assert robot.isselfcollided([0, 0]) is False
    collided.assert_not_called()


@pytest.mark.parametrize("depth", [1, 3])
def test_empty_spacer_links_are_transparent_to_adjacency(depth: int) -> None:
    base = Link(ET.Rz(), name="base", collision=[Sphere(1)])
    links = [base]
    for i in range(depth):
        links.append(
            Link(
                ET.tx(0),
                parent=links[-1],
                name=f"spacer{i}",
                collision=[CollisionShapeGroup([CollisionShapeGroup()])],
            )
        )
    links.append(Link(ET.tx(0), parent=links[-1], name="tip", collision=[Sphere(1)]))
    robot = Robot(links)
    with patch.object(Sphere, "iscollided") as collided:
        assert robot.isselfcollided([0]) is False
    collided.assert_not_called()


def test_links_in_same_gripper_are_not_checked() -> None:
    root = Link(ET.Rz(), name="root")
    hand = Link(ET.tx(0), parent=root, name="hand", collision=[Sphere(1)])
    left = Link(ET.tx(), parent=hand, name="left", collision=[Sphere(1)])
    right = Link(ET.ty(), parent=hand, name="right", collision=[Sphere(1)])
    robot = Robot([root, hand, left, right], gripper_links=hand)
    with patch.object(Sphere, "iscollided") as collided:
        assert robot.isselfcollided([0]) is False
    collided.assert_not_called()


def test_each_non_adjacent_shape_pair_is_checked_once() -> None:
    robot = chain()
    robot.links[0].collision.append(Sphere(0.1))
    robot.links[-1].collision.append(Sphere(0.1))
    with patch.object(Sphere, "iscollided", return_value=False) as collided:
        assert robot.isselfcollided([0, 0]) is False
    assert collided.call_count == 4


def test_first_collision_stops_the_search() -> None:
    robot = chain()
    robot.links[0].collision.append(Sphere(0.1))
    with patch.object(Sphere, "iscollided", return_value=True) as collided:
        assert robot.isselfcollided([0, np.pi]) is True
    collided.assert_called_once()


def test_skip_reuses_transforms() -> None:
    robot = chain()
    with (
        patch.object(robot, "_update_link_tf") as update_links,
        patch.object(robot, "update") as update,
        patch.object(robot, "fkine_all") as fkine_all,
        patch.object(Sphere, "iscollided", return_value=False),
    ):
        assert robot.isselfcollided([0, 0], skip=True) is False
    update_links.assert_not_called()
    update.assert_not_called()
    fkine_all.assert_not_called()


def test_self_argument_delegates_without_changing_external_shape_api() -> None:
    robot = chain()
    with patch.object(robot, "isselfcollided", return_value=True) as collided:
        assert robot.iscollided([0, np.pi], robot, skip=True) is True
    collided.assert_called_once_with([0, np.pi], skip=True)


@skip_no_collision_checking
@pytest.mark.parametrize("q,expected", [([0, 0], False), ([0, np.pi], True)])
def test_real_spheres_at_requested_configuration(q: ArrayLike, expected: bool) -> None:
    robot = chain()
    robot.q = [0.3, 0.4]
    assert robot.isselfcollided(q) is expected
    np.testing.assert_array_equal(robot.q, [0.3, 0.4])


@skip_no_collision_checking
def test_reported_urdf_panda_example_accepts_self() -> None:
    robot = models.Panda()
    before = robot.q.copy()
    assert isinstance(robot.iscollided(robot.q, robot), bool)
    np.testing.assert_array_equal(robot.q, before)


@skip_no_collision_checking
@pytest.mark.parametrize("model", [models.Panda, models.Frankie])
def test_panda_and_frankie_ready_pose_is_collision_free(model: type[Robot]) -> None:
    robot = model()
    before = robot.q.copy()
    gripper_configurations = [g.q.copy() for g in robot.grippers]
    assert robot.isselfcollided(robot.qr) is False
    assert robot.iscollided(robot.qr, robot) is False
    np.testing.assert_array_equal(robot.q, before)
    for gripper, configuration in zip(robot.grippers, gripper_configurations):
        np.testing.assert_array_equal(gripper.q, configuration)


@skip_no_collision_checking
def test_touching_non_adjacent_shapes_count_as_collision() -> None:
    robot = chain()
    robot.links[0].collision[0].radius = 1
    robot.links[-1].collision[0].radius = 1
    assert robot.isselfcollided([0, 0]) is True


@skip_no_collision_checking
def test_later_shape_on_link_can_be_the_only_collision() -> None:
    robot = chain()
    robot.links[-1].collision.append(Sphere(0.2, pose=SE3.Tx(-2)))
    assert robot.isselfcollided([0, 0]) is True


@skip_no_collision_checking
def test_nested_collision_group_preserves_local_transforms() -> None:
    robot = chain()
    group = CollisionShapeGroup([Sphere(0.2)])
    group.T = SE3.Tx(-2)
    robot.links[-1].collision = [group]
    before = group.T.copy()
    assert robot.isselfcollided([0, 0]) is True
    assert robot.isselfcollided([0, np.pi]) is False
    np.testing.assert_array_equal(group.T, before)


@skip_no_collision_checking
@pytest.mark.parametrize("group_type", [CollisionShapeGroup, SceneGroup])
def test_rotated_cuboid_in_nested_group(
    group_type: type[CollisionShapeGroup] | type[SceneGroup],
) -> None:
    base = Link(ET.Rz(), collision=[Sphere(0.1, pose=SE3.Tx(1))])
    middle = Link(ET.tx(0), parent=base, collision=[Sphere(0.1, pose=SE3.Tz(10))])
    group = group_type([Cuboid([3, 0.1, 0.1])])
    group.T = SE3.Rz(np.pi / 2)
    tip = Link(ET.Rz(), parent=middle, collision=[CollisionShapeGroup(), group])
    robot = Robot([base, middle, tip])
    # The long axis starts along y, clear of the sphere at x=1. Rotating
    # the joint by pi/2 brings that axis onto x and intersects the sphere.
    assert robot.isselfcollided([0, 0]) is False
    assert robot.isselfcollided([0, np.pi / 2]) is True
    assert robot.isselfcollided([0, 0]) is False


@skip_no_collision_checking
def test_collision_between_separate_grippers() -> None:
    root = Link(ET.Rz(), name="root")
    first = Link(ET.tx(1) * ET.tx(), parent=root, name="first", collision=[Sphere(0.2)])
    second = Link(
        ET.tx(2) * ET.tx(), parent=root, name="second", collision=[Sphere(0.2)]
    )
    robot = Robot([root, first, second], gripper_links=[first, second])
    assert robot.isselfcollided([0]) is False
    robot.grippers[1].q = [-1]
    assert robot.isselfcollided([0]) is True
    np.testing.assert_array_equal(robot.grippers[0].q, [0])
    np.testing.assert_array_equal(robot.grippers[1].q, [-1])


@skip_no_collision_checking
def test_default_configuration_and_repeated_queries() -> None:
    robot = chain()
    robot.q = [0, np.pi]
    assert robot.isselfcollided() is True
    assert robot.isselfcollided([0, 0]) is False
    # A public scene update must retain the requested configuration, rather
    # than overwrite collision poses with stale local link transforms.
    robot.update()
    assert robot.isselfcollided(skip=True) is False
    assert robot.isselfcollided([0, np.pi]) is True
    robot.update()
    assert robot.isselfcollided(skip=True) is True


@skip_no_collision_checking
@pytest.mark.parametrize(
    "q",
    [
        [0, 0],
        (0, 0),
        np.zeros(2, dtype=int),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 1)),
        None,
    ],
    ids=["list", "tuple", "integer-array", "float32-array", "column-vector", "current"],
)
def test_external_shape_queries_and_deprecated_alias(q: ArrayLike | None) -> None:
    robot = chain()
    assert robot.iscollided(q, Sphere(0.1)) is True
    assert robot.iscollided(q, Sphere(0.1, pose=SE3.Tx(5))) is False
    with pytest.warns(FutureWarning, match="deprecated"):
        assert robot.collided(q, Sphere(0.1)) is True
    assert robot.iscollided([0, np.pi], robot) is True
    with pytest.warns(FutureWarning, match="deprecated"):
        assert robot.collided([0, np.pi], robot) is True


@skip_no_collision_checking
def test_base_and_shape_local_transforms_are_applied() -> None:
    robot = chain()
    robot.base = SE3(3, -2, 1) * SE3.Ry(0.4)
    robot.links[-1].collision[0].T = SE3.Tz(1).A
    assert robot.isselfcollided([0, np.pi]) is False
    robot.links[-1].collision[0].T = SE3().A
    assert robot.isselfcollided([0, np.pi]) is True


@skip_no_collision_checking
def test_sibling_links_are_not_treated_as_adjacent() -> None:
    root = Link(ET.Rz(), name="root")
    left = Link(ET.tx(1), parent=root, name="left", collision=[Sphere(0.2)])
    right = Link(
        ET.ty(0.1) * ET.Rz(),
        parent=root,
        name="right",
        collision=[Sphere(0.2, pose=SE3.Tx(1))],
    )
    robot = Robot([root, left, right])
    assert robot.isselfcollided([0, 0]) is True
    assert robot.isselfcollided([0, np.pi]) is False


@skip_no_collision_checking
def test_gripper_links_use_their_own_configuration() -> None:
    base = Link(ET.Rz(), name="base", collision=[Sphere(0.2)])
    wrist = Link(
        ET.tx(1), parent=base, name="wrist", collision=[Sphere(0.1, pose=SE3.Tz(10))]
    )
    finger = Link(
        ET.tx(1) * ET.tx(), parent=wrist, name="finger", collision=[Sphere(0.2)]
    )
    robot = Robot([base, wrist, finger], gripper_links=finger)
    assert robot.isselfcollided([0]) is False
    robot.grippers[0].q = [-2]
    assert robot.isselfcollided([0]) is True


@skip_no_collision_checking
def test_dh_robot_with_collision_geometry() -> None:
    robot = DHRobot(
        [
            RevoluteDH(a=1, collision=[Sphere(0.2, pose=SE3.Tx(-1))]),
            RevoluteDH(a=1, collision=[Sphere(0.1, pose=SE3.Tz(10))]),
            RevoluteDH(a=0, collision=[Sphere(0.2)]),
        ]
    )
    assert robot.isselfcollided([0, 0, 0]) is False
    assert robot.isselfcollided([0, np.pi, 0]) is True


@pytest.mark.parametrize(
    "q", [[0], [0, 0, 0], [[0, 0], [0, 0]], [0, np.nan], [np.inf, 0]]
)
def test_configuration_must_be_one_joint_vector(q: ArrayLike) -> None:
    with pytest.raises(ValueError):
        chain().isselfcollided(q)
