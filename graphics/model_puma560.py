# from graphics.graphics_robot import *
# from graphics.graphics_stl import *
from graphics.graphics_robot import GraphicalRobot, RotationalJoint, StaticJoint, Gripper
from vpython import vector
from spatialmath import SE3


def import_puma_560():
    """
    Create a Robot class object based on the puma560 robot

    :return: Puma560 robot
    :rtype: class:`graphics.graphics_robot.GraphicalRobot`
    """
    puma560 = GraphicalRobot()

    puma560.append_made_link(create_link_0())
    puma560.append_made_link(create_link_1())
    puma560.append_made_link(create_link_2())
    puma560.append_made_link(create_link_3())
    puma560.append_made_link(create_link_4())
    puma560.append_made_link(create_link_5())
    puma560.append_made_link(create_link_6())

    # TODO set poses correctly to reflect 'L' shape default position

    return puma560


def create_link_0():
    """
    Create the specific joint link and return it as a Joint object

    :return: Rotational joint representing joint 0
    :rtype: class:`graphics.graphics_robot.RotationalJoint
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link0.stl'

    link = RotationalJoint(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj_z_origin = stl_obj.pos.z - stl_obj.width / 2
    stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, stl_obj_z_origin)
    stl_obj_required_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, 0)
    link.set_stl_joint_origin(stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.blue

    return link


def create_link_1():
    """
    Create the specific joint link and return it as a Joint object

    :return: Rotational joint representing joint 1
    :rtype: class:`graphics.graphics_robot.StaticJoint
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link1.stl'

    link = StaticJoint(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj_y_origin = -stl_obj.height / 2
    stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj_y_origin, stl_obj.pos.z)
    stl_obj_required_origin_location = vector(stl_obj.pos.x, 0, stl_obj.pos.z)
    link.set_stl_joint_origin(stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.green

    return link


def create_link_2():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 2
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link2.stl'

    link = RotationalJoint(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj_x_origin = -0.437
    stl_obj_z_origin = 0.15
    stl_obj_current_origin_location = vector(stl_obj_x_origin, stl_obj.pos.y, stl_obj_z_origin)
    stl_obj_required_origin_location = vector(0, stl_obj.pos.y, 0)
    link.set_stl_joint_origin(stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.red

    return link


def create_link_3():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 3
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link3.stl'

    link = RotationalJoint(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj_y_origin = -stl_obj.height / 2
    stl_obj_z_origin = -0.05
    stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj_y_origin, stl_obj_z_origin)
    stl_obj_required_origin_location = vector(stl_obj.pos.x, 0, 0)
    link.set_stl_joint_origin(stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.cyan

    return link


def create_link_4():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 4
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link4.stl'

    link = RotationalJoint(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    # stl_obj.rotate(angle=radians(-90), axis=z_axis_vector, origin=vector(0, 0, 0))
    # stl_obj_x_origin = -0.071
    # stl_obj_current_origin_location = vector(stl_obj_x_origin, stl_obj.pos.y, stl_obj.pos.z)
    # stl_obj_required_origin_location = vector(0, stl_obj.pos.y, stl_obj.pos.z)
    # stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.magenta

    return link


def create_link_5():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 5
        :rtype: class:`graphics.graphics_robot.RotationalJoint
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link5.stl'

    link = RotationalJoint(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    # stl_obj.rotate(angle=radians(90), axis=x_axis_vector, origin=vector(0, 0, 0))
    # stl_obj.rotate(angle=radians(90), axis=z_axis_vector, origin=vector(0, 0, 0))
    # stl_obj_current_origin_location = vector(0, 0, 0)
    # stl_obj_required_origin_location = vector(0, 0, 0)
    # stl_obj = set_stl_origin(stl_obj, stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.yellow

    return link


def create_link_6():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 6
        :rtype: class:`graphics.graphics_robot.Gripper
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link6.stl'

    link = Gripper(SE3(), stl_obj_path)
    stl_obj = link.get_graphic_object()

    # Orient the object so that it's origin and toolpoint in known locations
    # This way, rotations are relative to the correct 3D position of the object
    stl_obj_z_origin = 0.043
    stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, stl_obj_z_origin)
    stl_obj_required_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, 0)
    link.set_stl_joint_origin(stl_obj_current_origin_location, stl_obj_required_origin_location)

    # Change color
    # stl_obj.color = color.black

    return link
