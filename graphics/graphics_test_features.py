from graphics.model_puma560 import *


def test_grid():
    canvas_grid = init_canvas()
    while True:
        sleep(1)
        canvas_grid.update_grid()
        print("XY:", canvas_grid.grid_object[0].pos)


def test_reference_frames():
    le = 0.2
    canvas_grid = init_canvas()

    # Test 1 | Position (0, 0, 0), Axis (1, 0, 0), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(0, 0, 0), vector(1, 0, 0), radians(0))
    # Actual
    #arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), length=le, color=color.purple)

    # Test 2 | Position (1, 1, 1), Axis (0, 0, 1), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(1, 1, 1), vector(0, 0, 1), radians(0))
    # Actual
    #arrow(pos=vector(1, 1, 1), axis=vector(0, 0, 1), length=le, color=color.purple)

    # Test 3 | Position (2, 2, 2), Axis (1, 0, 0), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(2, 2, 2), vector(1, 0, 0), radians(30))
    # Actual
    #arrow(pos=vector(2, 2, 2), axis=vector(1, 0, 0), length=le, color=color.purple).rotate(radians(30))

    # Test 4 | Position (3, 3, 3), Axis (1, 1, 1), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(3, 3, 3), vector(1, 1, 1), radians(0))
    # Actual
    #arrow(pos=vector(3, 3, 3), axis=vector(1, 1, 1), length=le, color=color.purple)

    # Test 5 | Position (4, 4, 4), Axis (1, 1, 1), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(4, 4, 4), vector(1, 1, 1), radians(30))
    # Actual
    #arrow(pos=vector(4, 4, 4), axis=vector(1, 1, 1), length=le, color=color.purple).rotate(radians(30))

    # Test 6 | Position (5, 5, 5), Axis (2, -1, 4), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(5, 5, 5), vector(2, -1, 4), radians(0))
    # Actual
    #arrow(pos=vector(5, 5, 5), axis=vector(2, -1, 4), length=le, color=color.purple)

    # Test 7 | Position (6, 6, 6), Axis (2, -1, 4), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(6, 6, 6), vector(2, -1, 4), radians(30))
    # Actual
    #arrow(pos=vector(6, 6, 6), axis=vector(2, -1, 4), length=le, color=color.purple).rotate(radians(30))


def test_import_stl():
    canvas_grid = init_canvas()

    puma560 = import_puma_560()
    puma560.move_base(vector(1, 1, 0))

    """
    puma560.set_reference_visibility(False)
    puma560.print_joint_angles(True)

    sleep(2)
    puma560.set_joint_angle(4, radians(35))

    sleep(2)
    puma560.set_joint_angle(2, radians(-56))

    sleep(2)
    puma560.set_joint_angle(0, radians(78))

    sleep(2)
    puma560.set_all_joint_angles([
        0, 0, 0, 0, 0, 0, 0
    ])

    
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
    

    puma560.print_joint_angles(True)
    """


def test_rotational_link():
    canvas_grid = init_canvas()
    # rot = vector(0, 0, 1)
    rot = vector(0, 1, 0)
    # rot = vector(1, 0, 0)
    rot_link = RotationalJoint(vector(0, 0, 0), vector(1, 0, 0), x_axis=x_axis_vector, rotation_axis=rot)
    rot_link.rotate_around_joint_axis(radians(30), x_axis_vector)
    rot_link.draw_reference_frame(True)

    # for angle in [0, 45, 90, 135, 180, -135, -90, -45, 33, -66, -125, 162, 360, 450, -270, -333]:
    #    sleep(5)
    for angle in range(0, 360):
        sleep(0.05)
        rot_link.rotate_joint(radians(angle))


def test_graphical_robot():
    canvas_grid = init_canvas()

    x = GraphicalRobot([
        RotationalJoint(vector(0, 0, 0), vector(1, 1, 1))
    ])

    x.set_joint_angle(0, radians(20))


def test_place_joint():
    pass


def test_animate_joints():
    pass


def test_import_textures():
    pass
