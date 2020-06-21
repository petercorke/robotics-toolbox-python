from graphics.model_puma560 import *


def test_grid_updating():
    """
    This test will create a canvas and update the grid every second.
    Eventually, the grid will only update through callbacks of key/button releases.
    """
    canvas_grid = init_canvas()
    while True:
        sleep(1)
        canvas_grid.update_grid()


def test_reference_frame_pose():
    """
    This test will create a canvas, and place reference frames at the given positions and orientations.
    Each frame must be manually inspected for validity.
    """
    canvas_grid = init_canvas()

    # Test 1 | Position (0, 0, 0), Axis (1, 0, 0), No Rotation
    draw_reference_frame_axes(vector(0, 0, 0), vector(1, 0, 0), radians(0))

    # Test 2 | Position (1, 1, 1), Axis (0, 0, 1), No Rotation
    draw_reference_frame_axes(vector(1, 1, 1), vector(0, 0, 1), radians(0))

    # Test 3 | Position (2, 2, 2), Axis (1, 0, 0), 30 deg rot
    draw_reference_frame_axes(vector(2, 2, 2), vector(1, 0, 0), radians(30))

    # Test 4 | Position (3, 3, 3), Axis (1, 1, 1), No Rotation
    draw_reference_frame_axes(vector(3, 3, 3), vector(1, 1, 1), radians(0))

    # Test 5 | Position (4, 4, 4), Axis (1, 1, 1), 30 deg rot
    draw_reference_frame_axes(vector(4, 4, 4), vector(1, 1, 1), radians(30))

    # Test 6 | Position (5, 5, 5), Axis (2, -1, 4), No Rotation
    draw_reference_frame_axes(vector(5, 5, 5), vector(2, -1, 4), radians(0))

    # Test 7 | Position (6, 6, 6), Axis (2, -1, 4), 30 deg rot
    draw_reference_frame_axes(vector(6, 6, 6), vector(2, -1, 4), radians(30))


def test_import_stl():
    """
    This test will create a canvas with the Puma560 model loaded in.
    The robot should have all joint angles to 0 (Robot is in an 'L' shape from (1, 1, 0) in the +X axis direction)
    """
    canvas_grid = init_canvas()

    puma560 = import_puma_560()
    puma560.move_base(vector(1, 1, 0))


def test_rotational_link():
    """
    This test will create a simple rotational link from (0, 0, 0) to (1, 0, 0).
    Depending on which for loop is commented out:
    The joint will then rotate in a positive direction about it's +y axis.
    OR
    The joint will rotate to the given angles in the list.
    """
    canvas_grid = init_canvas()
    # rot = vector(0, 0, 1)
    rot = vector(0, 1, 0)
    # rot = vector(1, 0, 0)
    rot_link = RotationalJoint(vector(0, 0, 0), vector(1, 0, 0), x_axis=x_axis_vector, rotation_axis=rot)
    rot_link.rotate_around_joint_axis(radians(30), x_axis_vector)
    rot_link.draw_reference_frame(True)

    # for angle in [0, 45, 90, 135, 180, -135, -90, -45, 33, -66, -125, 162, 360, 450, -270, -333]:
    #    sleep(5)
    #    rot_link.rotate_joint(radians(angle))

    for angle in range(0, 360):
        sleep(0.05)
        rot_link.rotate_joint(radians(angle))


def test_graphical_robot_creation():
    """
    This test will create a simple 3-link graphical robot.
    The joints are then set to particular angles to show rotations.
    """
    canvas_grid = init_canvas()

    x = GraphicalRobot([
        RotationalJoint(vector(0, 0, 0), vector(1, 0, 0)),
        RotationalJoint(vector(1, 0, 0), vector(2, 0, 0)),
        RotationalJoint(vector(2, 0, 0), vector(3, 0, 0))
    ])

    sleep(2)
    x.set_all_joint_angles([radians(-45), radians(45), radians(15)])


def test_puma560_angle_change():
    """
    This test loads in the Puma560 model and changes its angles over time.
    Joint angles are printed for validation.
    """
    canvas_grid = init_canvas()

    puma560 = import_puma_560()
    puma560.move_base(vector(1, 1, 0))

    puma560.set_reference_visibility(False)
    print("Prior Angles")
    puma560.print_joint_angles(True)

    sleep(2)
    puma560.set_all_joint_angles([
        radians(45), 0, 0, 0, 0, 0, 0,
    ])

    sleep(2)
    puma560.set_all_joint_angles([
        radians(45), 0, radians(-90), 0, 0, 0, 0,
    ])

    sleep(2)
    puma560.set_joint_angle(4, radians(156))

    sleep(2)
    puma560.set_joint_angle(2, radians(-23))

    print("Final Angles")
    puma560.print_joint_angles(True)


def test_animate_joints():
    pass


def test_import_textures():
    pass
