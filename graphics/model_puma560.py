from graphics.graphics_robot import *
from graphics.graphics_stl import *


def import_puma_560():

    puma560 = Robot(
        [
            create_link_0(),
            create_link_1(),
            create_link_2(),
            create_link_3(),
            create_link_4(),
            create_link_5(),
            create_link_6()
        ]
    )

    puma560.set_reference_visibility(False)

    return puma560


def create_link_0():
    # Load the STL file into an object
    stl_obj = import_object_from_stl(filename='link0')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj_z_origin = stl_obj.pos.z - stl_obj.width / 2
    stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, stl_obj_z_origin)
    stl_obj_required_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, 0)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.blue

    # Create the robot link
    connection_from = vector(0, 0, 0)
    connection_to = vector(0, 0, stl_obj.width)
    link = RotationalJoint(connection_from,
                           connection_to,
                           x_axis=vector(1, 0, 0),
                           rotation_axis=vector(0, 0, 1),
                           graphic_obj=stl_obj)
    return link


def create_link_1():
    # Load the STL file into an object
    stl_obj = import_object_from_stl(filename='link1')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=vector(0, 1, 0), origin=vector(0, 0, 0))
    stl_obj.rotate(angle=radians(90), axis=vector(0, 0, 1), origin=vector(0, 0, 0))
    stl_obj.rotate(angle=radians(90), axis=vector(0, 1, 0), origin=vector(0, 0, 0))
    stl_obj_z_origin = -stl_obj.height / 2
    stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, stl_obj_z_origin)
    stl_obj_required_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, 0)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.green

    # Create the robot link
    connection_from = vector(0, 0, 0)
    # Numbers come from previous adjustments, plus extra observed in meshlab
    connection_to = vector(0, 0.184, stl_obj.width / 2)
    link = StaticJoint(connection_from,
                       connection_to,
                       x_axis=vector(1, 0, 0),
                       graphic_obj=stl_obj)
    return link


def create_link_2():
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link2')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(-90), axis=vector(1, 0, 0), origin=vector(0, 0, 0))
    stl_obj_x_origin = -0.437
    stl_obj_y_origin = 0.15
    stl_obj_current_origin_location = vector(stl_obj_x_origin, stl_obj_y_origin, stl_obj.pos.z)
    stl_obj_required_origin_location = vector(0, 0, stl_obj.pos.z)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.red

    # Create the robot link
    connection_from = vector(0, 0, 0)
    # Numbers come from previous adjustments, plus extra observed in meshlab
    connection_to = vector(0.437, 0.0338, 0)
    link = RotationalJoint(connection_from,
                           connection_to,
                           x_axis=vector(1, 0, 0),
                           rotation_axis=vector(0, 1, 0),
                           graphic_obj=stl_obj)
    return link


def create_link_3():
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link3')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=vector(0, 1, 0), origin=vector(0, 0, 0))
    stl_obj_y_origin = stl_obj.height / 2
    stl_obj_x_origin = -0.05
    stl_obj_current_origin_location = vector(stl_obj_x_origin, stl_obj_y_origin, stl_obj.pos.z)
    stl_obj_required_origin_location = vector(0, 0, stl_obj.pos.z)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.cyan

    # Create the robot link
    connection_from = vector(0, 0, 0)
    # Numbers come from previous adjustments, plus extra observed in meshlab
    connection_to = vector(0.36 + 0.05, -stl_obj.height / 2, 0)
    link = RotationalJoint(connection_from,
                           connection_to,
                           x_axis=vector(1, 0, 0),
                           rotation_axis=vector(0, 1, 0),
                           graphic_obj=stl_obj)
    return link


def create_link_4():
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link4')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(-90), axis=vector(0, 0, 1), origin=vector(0, 0, 0))
    stl_obj_x_origin = -0.071
    stl_obj_current_origin_location = vector(stl_obj_x_origin, stl_obj.pos.y, stl_obj.pos.z)
    stl_obj_required_origin_location = vector(0, stl_obj.pos.y, stl_obj.pos.z)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.magenta

    # Create the robot link
    connection_from = vector(0, 0, 0)
    # Numbers come from previous adjustments, plus extra observed in meshlab
    connection_to = vector(0.071, 0, 0)
    link = RotationalJoint(connection_from,
                           connection_to,
                           x_axis=vector(1, 0, 0),
                           rotation_axis=vector(1, 0, 0),
                           graphic_obj=stl_obj)
    return link


def create_link_5():
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link5')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=vector(1, 0, 0), origin=vector(0, 0, 0))
    stl_obj.rotate(angle=radians(90), axis=vector(0, 0, 1), origin=vector(0, 0, 0))
    stl_obj_current_origin_location = vector(0, 0, 0)
    stl_obj_required_origin_location = vector(0, 0, 0)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.yellow

    # Create the robot link
    connection_from = vector(0, 0, 0)
    # Numbers come from previous adjustments, plus extra observed in meshlab
    connection_to = vector(0.046, 0, 0)
    link = RotationalJoint(connection_from,
                           connection_to,
                           x_axis=vector(1, 0, 0),
                           rotation_axis=vector(0, 0, 1),
                           graphic_obj=stl_obj)
    return link


def create_link_6():
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link6')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=vector(0, 1, 0), origin=vector(0, 0, 0))
    stl_obj_x_origin = 0.043
    stl_obj_current_origin_location = vector(stl_obj_x_origin, stl_obj.pos.y, stl_obj.pos.z)
    stl_obj_required_origin_location = vector(0, stl_obj.pos.y, stl_obj.pos.z)
    stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)
    # Change color
    stl_obj.color = color.black

    # Create the robot link
    connection_from = vector(0, 0, 0)
    # Numbers come from previous adjustments, plus extra observed in meshlab
    connection_to = vector(stl_obj.length, 0, 0)
    link = Gripper(connection_from,
                   connection_to,
                   x_axis=vector(1, 0, 0),
                   graphic_obj=stl_obj)
    return link
