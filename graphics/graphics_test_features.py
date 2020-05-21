from graphics.graphics_robot import *


def test_grid():
    canvas_grid = init_canvas()
    while True:
        sleep(1)
        canvas_grid.update_grid()


def test_reference_frames():
    le = 0.8
    canvas_grid = init_canvas()

    # Test 1 | Position (0, 0, 0), Axis (1, 0, 0), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(0, 0, 0), vector(1, 0, 0), radians(0))
    # Actual
    arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), length=le, color=color.purple)

    # Test 2 | Position (1, 1, 1), Axis (0, 0, 1), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(1, 1, 1), vector(0, 0, 1), radians(0))
    # Actual
    arrow(pos=vector(1, 1, 1), axis=vector(0, 0, 1), length=le, color=color.purple)

    # Test 3 | Position (2, 2, 2), Axis (1, 0, 0), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(2, 2, 2), vector(1, 0, 0), radians(30))
    # Actual
    arrow(pos=vector(2, 2, 2), axis=vector(1, 0, 0), length=le, color=color.purple).rotate(radians(30))

    # Test 4 | Position (3, 3, 3), Axis (1, 1, 1), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(3, 3, 3), vector(1, 1, 1), radians(0))
    # Actual
    arrow(pos=vector(3, 3, 3), axis=vector(1, 1, 1), length=le, color=color.purple)

    # Test 5 | Position (4, 4, 4), Axis (1, 1, 1), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(4, 4, 4), vector(1, 1, 1), radians(30))
    # Actual
    arrow(pos=vector(4, 4, 4), axis=vector(1, 1, 1), length=le, color=color.purple).rotate(radians(30))

    # Test 6 | Position (5, 5, 5), Axis (2, -1, 4), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(5, 5, 5), vector(2, -1, 4), radians(0))
    # Actual
    arrow(pos=vector(5, 5, 5), axis=vector(2, -1, 4), length=le, color=color.purple)

    # Test 7 | Position (6, 6, 6), Axis (2, -1, 4), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(6, 6, 6), vector(2, -1, 4), radians(30))
    # Actual
    arrow(pos=vector(6, 6, 6), axis=vector(2, -1, 4), length=le, color=color.purple).rotate(radians(30))


def test_import_stl():
    canvas_grid = init_canvas()
    """
    robot0 = import_object_from_stl(filename='link0')
    robot0_z_origin = robot0.pos.z - robot0.width / 2
    robot0_current_origin_location = vector(robot0.pos.x, robot0.pos.y, robot0_z_origin)
    robot0_required_origin_location = vector(robot0.pos.x, robot0.pos.y, 0)
    robot0 = set_stl_origin(robot0, robot0_current_origin_location, robot0_required_origin_location)
    robot0.color = color.blue
    robot0.visible = False

    robot1 = import_object_from_stl(filename='link1')
    robot1.rotate(angle=radians(90), axis=vector(0, 1, 0), origin=vector(0, 0, 0))
    robot1.rotate(angle=radians(90), axis=vector(1, 0, 0), origin=vector(0, 0, 0))
    robot1_z_origin = -robot1.height/2
    robot1_current_origin_location = vector(robot1.pos.x, robot1.pos.y, robot1_z_origin)
    robot1_required_origin_location = vector(robot1.pos.x, robot1.pos.y, 0)
    robot1 = set_stl_origin(robot1, robot1_current_origin_location, robot1_required_origin_location)
    robot1.color = color.green
    robot1.visible = False
    
    robot2 = import_object_from_stl('link2')
    robot2.rotate(angle=radians(-90), axis=vector(1, 0, 0), origin=vector(0, 0, 0))
    robot2_x_origin = -0.437
    robot2_y_origin = 0.15
    robot2_current_origin_location = vector(robot2_x_origin, robot2_y_origin, robot2.pos.z)
    robot2_required_origin_location = vector(0, 0, robot2.pos.z)
    robot2 = set_stl_origin(robot2, robot2_current_origin_location, robot2_required_origin_location)
    robot2.color = color.red
    """
    #robot3 = import_object_from_stl('link3')
    #robot3.color = color.cyan
    if 0:
        robot4 = import_object_from_stl('link4')
        robot4.color = color.magenta
        robot4.pos += vector(4, 0, 0)

        robot5 = import_object_from_stl('link5')
        robot5.color = color.yellow
        robot5.pos += vector(5, 0, 0)

        robot6 = import_object_from_stl('link6')
        robot6.color = color.black
        robot6.pos += vector(6, 0, 0)


def test_rotational_link():
    canvas_grid = init_canvas()
    rot_link = RotationalJoint(vector(1, 1, 1), vector(0.3, 3, 3))
    rot_link.draw_reference_frame(True)

    for angle in [0, 45, 90, 135, 180, -135, -90, -45, 33, -66, -125, 162]:
        sleep(5)
        rot_link.rotate_joint(radians(angle))


def test_place_joint():
    pass


def test_animate_joints():
    pass


def test_import_textures():
    pass
