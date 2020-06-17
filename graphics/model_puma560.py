from graphics.graphics_robot import *
from graphics.graphics_stl import *


def import_puma_560():
    """
    Create a Robot class object based on the puma560 robot

    :return: Puma560 robot
    :rtype: class:`graphics.graphics_robot.GraphicalRobot`
    """
    puma560 = GraphicalRobot(
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
    return puma560


def create_link_0():
    """
    Create the specific joint link and return it as a Joint object

    :return: Rotational joint representing joint 0
    :rtype: class:`graphics.graphics_robot.RotationalJoint
    """
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
                           x_axis=x_axis_vector,
                           rotation_axis=z_axis_vector,
                           graphic_obj=stl_obj)
    return link


def create_link_1():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 1
        :rtype: class:`graphics.graphics_robot.StaticJoint
        """
    # Load the STL file into an object
    stl_obj = import_object_from_stl(filename='link1')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=y_axis_vector, origin=vector(0, 0, 0))
    stl_obj.rotate(angle=radians(90), axis=z_axis_vector, origin=vector(0, 0, 0))
    stl_obj.rotate(angle=radians(90), axis=y_axis_vector, origin=vector(0, 0, 0))
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
                       x_axis=x_axis_vector,
                       graphic_obj=stl_obj)
    return link


def create_link_2():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 2
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link2')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(-90), axis=x_axis_vector, origin=vector(0, 0, 0))
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
                           x_axis=x_axis_vector,
                           rotation_axis=y_axis_vector,
                           graphic_obj=stl_obj)
    return link


def create_link_3():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 3
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link3')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=y_axis_vector, origin=vector(0, 0, 0))
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
                           x_axis=x_axis_vector,
                           rotation_axis=y_axis_vector,
                           graphic_obj=stl_obj)
    return link


def create_link_4():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 4
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link4')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(-90), axis=z_axis_vector, origin=vector(0, 0, 0))
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
                           x_axis=x_axis_vector,
                           rotation_axis=x_axis_vector,
                           graphic_obj=stl_obj)
    return link


def create_link_5():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 5
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link5')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=x_axis_vector, origin=vector(0, 0, 0))
    stl_obj.rotate(angle=radians(90), axis=z_axis_vector, origin=vector(0, 0, 0))
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
                           x_axis=x_axis_vector,
                           rotation_axis=z_axis_vector,
                           graphic_obj=stl_obj)
    return link


def create_link_6():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 6
        :rtype: class:`graphics.graphics_robot.Gripper
        """
    # Load the STL file into an object
    stl_obj = import_object_from_stl('link6')
    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj.rotate(angle=radians(90), axis=y_axis_vector, origin=vector(0, 0, 0))
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
                   x_axis=x_axis_vector,
                   graphic_obj=stl_obj)
    return link
