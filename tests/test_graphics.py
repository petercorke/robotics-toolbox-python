#!/usr/bin/env python3
"""
These functions are not ordinary testing functions.
These tests cannot be automated, and must be manually validated.

To execute, import from this file into the console. "import test.test_graphics as test_gph"
Next select which test you which to run, and call the function.
A canvas will be created and display the respective graphics.
Verify the output is as expected.
Then close the browser window and run a different function. (Help clear graphics. Currently, no clearing implemented).

Alternatively, executing this file will run the test_puma560_angle_change() function.
"""
from time import sleep
from numpy import array
from spatialmath import SE3
import graphics as gph


def test_grid_updating():
    """
    This test will create a canvas and update the grid every second.
    Eventually, the grid will only update through callbacks of key/button releases.
    """
    canvas_grid = gph.init_canvas()
    while True:
        sleep(1)
        canvas_grid.update_grid()


def test_reference_frame_pose():
    """
    This test will create a canvas, and place reference frames at the given positions and orientations.
    Each frame must be manually inspected for validity.
    """
    canvas_grid = gph.init_canvas()

    # Test 1 | Position (0, 0, 0), Axis (1, 0, 0), No Rotation
    arr = array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    gph.draw_reference_frame_axes(SE3(arr))

    # Test 2 | Position (1, 1, 1), Axis (0, 0, 1), No Rotation
    arr = array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 1]
    ])
    gph.draw_reference_frame_axes(SE3(arr))

    # Test 3 | Position (2, 2, 2), Axis (0, 1, 0), 30 deg rot
    arr = array([
        [0, -1, 0, 2],
        [1, 0, 0, 2],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ])
    se = SE3(arr)
    gph.draw_reference_frame_axes(se.Rx(30, 'deg'))


def test_import_stl():
    """
    This test will create a canvas with the Puma560 model loaded in.
    """
    canvas_grid = gph.init_canvas()
    puma560 = gph.import_puma_560()


def test_rotational_link():
    """
    This test will create a simple rotational link from (0, 0, 0) to (1, 0, 0).
    """
    canvas_grid = gph.init_canvas()

    arr = array([
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [1, 0,  0, 0],
        [0, 0,  0, 1]
    ])
    se3 = SE3(arr)

    rot_link = gph.RotationalJoint(se3, 1.0)
    rot_link.draw_reference_frame(True)


def test_graphical_robot_creation():
    """
    This test will create a simple 3-link graphical robot.
    The joints are then set to new poses to show updating does occur.
    """
    canvas_grid = gph.init_canvas()

    p = SE3()

    p1 = p
    p2 = p.Tx(1)
    p3 = p.Tx(2)

    robot = gph.GraphicalRobot()

    robot.append_link('r', p1, 1.0)
    robot.append_link('R', p2, 1.0)
    robot.append_link('r', p3, 1.0)

    arr = array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    new_p1 = SE3(arr)

    arr = array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1]
    ])
    new_p2 = SE3(arr)

    arr = array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 2],
        [0, 0, 0, 1]
    ])
    new_p3 = SE3(arr)

    sleep(2)

    robot.set_joint_poses([
        new_p1,
        new_p2,
        new_p3
    ])


# TODO
def test_puma560_angle_change():
    """
    This test loads in the Puma560 model and changes its angles over time.
    Joint angles are printed for validation.
    """
    canvas_grid = gph.init_canvas()
    puma560 = gph.import_puma_560()

    puma560.set_reference_visibility(False)

    print("Prior Poses")
    puma560.print_joint_poses()

    puma560.set_joint_poses([
        # TODO
    ])

    print("Final Poses")
    puma560.print_joint_angles()


def test_clear_scene():
    """
    This test will import the Puma560 model, then after 2 seconds, clear the canvas of all models.
    """
    the_grid = gph.init_canvas()

    puma560 = gph.import_puma_560()
    puma560.set_reference_visibility(True)

    sleep(2)

    the_grid.clear_scene()
    del puma560


# TODO
def test_clear_scene_with_grid_updating():
    """
    This test will import the Puma560 model, then after 2 seconds, clear the canvas of all models.
    Meanwhile, grid update calls have been placed in between. (Currently the only way to update the grid)
    """
    the_grid = gph.init_canvas()
    puma560 = gph.import_puma_560()
    the_grid.update_grid()

    puma560.set_joint_poses([
        # TODO
    ])
    the_grid.update_grid()

    sleep(2)
    the_grid.update_grid()

    the_grid.clear_scene()
    del puma560

    while True:
        sleep(1)
        the_grid.update_grid()


# TODO
def test_animate_joints():
    """
    This test will create a three link robot, and iterate through a series of frames to animate movement.
    """
    the_grid = gph.init_canvas()

    p = SE3()

    p1 = p
    p2 = p.Tx(1)
    p3 = p.Tx(2)

    robot = gph.GraphicalRobot()

    robot.append_link('r', p1, 1.0)
    robot.append_link('R', p2, 1.0)
    robot.append_link('r', p3, 1.0)

    robot.animate([
        [SE3().Rand(), SE3().Rand(), SE3().Rand()],
        [SE3().Rand(), SE3().Rand(), SE3().Rand()],
        [SE3().Rand(), SE3().Rand(), SE3().Rand()],
        [SE3().Rand(), SE3().Rand(), SE3().Rand()],
        [SE3().Rand(), SE3().Rand(), SE3().Rand()],
        [SE3().Rand(), SE3().Rand(), SE3().Rand()],
        [SE3().Rand(), SE3().Rand(), SE3().Rand()]
    ], 7)


def test_import_textures():
    pass


if __name__ == "__main__":
    # run the Puma demo by default
    test_puma560_angle_change()
