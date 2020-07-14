from graphics.graphics_robot import GraphicalRobot, RotationalJoint, StaticJoint, Gripper
from roboticstoolbox import Puma560
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

    # Get the poses for a zero-position
    puma = Puma560()
    poses = puma.fkine(puma.config('qz'), alltout=True)

    puma560.set_joint_poses([
        SE3(),  # 0 (Base doesn't change)
        poses[0],  # 1
        poses[1],  # 2
        poses[2],  # 3
        poses[3],  # 4
        poses[4],  # 5
        poses[5]   # 6
    ])

    return puma560


def create_link_0():
    """
    Create the specific joint link and return it as a Joint object

    :return: Rotational joint representing joint 0
    :rtype: class:`graphics.graphics_robot.StaticJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link0.stl'

    link = StaticJoint(SE3(), stl_obj_path)

    # Change color
    # stl_obj.color = color.blue

    return link


def create_link_1():
    """
    Create the specific joint link and return it as a Joint object

    :return: Rotational joint representing joint 1
    :rtype: class:`graphics.graphics_robot.RotationalJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link1.stl'

    link = RotationalJoint(SE3(), stl_obj_path)

    # Change color
    # stl_obj.color = color.green

    return link


def create_link_2():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 2
        :rtype: class:`graphics.graphics_robot.RotationalJoint`
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link2.stl'

    link = RotationalJoint(SE3(), stl_obj_path)

    # Change color
    # stl_obj.color = color.red

    return link


def create_link_3():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 3
        :rtype: class:`graphics.graphics_robot.RotationalJoint`
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link3.stl'

    link = RotationalJoint(SE3(), stl_obj_path)
    # stl_obj = link.get_graphic_object()

    # Change color
    # stl_obj.color = color.cyan

    return link


def create_link_4():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 4
        :rtype: class:`graphics.graphics_robot.RotationalJoint`
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link4.stl'

    link = RotationalJoint(SE3(), stl_obj_path)

    # Change color
    # stl_obj.color = color.magenta

    return link


def create_link_5():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 5
        :rtype: class:`graphics.graphics_robot.RotationalJoint`
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link5.stl'

    link = RotationalJoint(SE3(), stl_obj_path)

    # Change color
    # stl_obj.color = color.yellow

    return link


def create_link_6():
    """
        Create the specific joint link and return it as a Joint object

        :return: Rotational joint representing joint 6
        :rtype: class:`graphics.graphics_robot.Gripper`
        """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link6.stl'

    link = Gripper(SE3(), stl_obj_path)

    # Change color
    # stl_obj.color = color.black

    return link
