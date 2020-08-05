from graphics.graphics_robot import GraphicalRobot, RotationalJoint, StaticJoint, Gripper
from roboticstoolbox.models.Puma560 import Puma560
from spatialmath import SE3


def import_puma_560(g_canvas):
    """
    Create a Robot class object based on the puma560 robot

    :param g_canvas: The canvas to display the robot in
    :type g_canvas: class:`graphics.graphics_canvas.GraphicsCanvas`
    :return: Puma560 robot
    :rtype: class:`graphics.graphics_robot.GraphicalRobot`
    """
    puma560 = GraphicalRobot(g_canvas, 'Puma560')

    puma560.append_made_link(create_link_0(g_canvas.scene))
    puma560.append_made_link(create_link_1(g_canvas.scene))
    puma560.append_made_link(create_link_2(g_canvas.scene))
    puma560.append_made_link(create_link_3(g_canvas.scene))
    puma560.append_made_link(create_link_4(g_canvas.scene))
    puma560.append_made_link(create_link_5(g_canvas.scene))
    puma560.append_made_link(create_link_6(g_canvas.scene))

    # Get the poses for a zero-position
    puma = Puma560()
    poses = puma.fkine(puma.qz, alltout=True)

    puma560.set_joint_poses([
        poses[0],
        poses[1],
        poses[2],
        poses[3],
        poses[4],
        poses[5],
        poses[6]
    ])

    return puma560


def create_link_0(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 0
    :rtype: class:`graphics.graphics_robot.StaticJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link0.stl'

    link = StaticJoint(SE3(), stl_obj_path, scene)

    # Change color
    link.set_texture(colour=[0, 0, 1])

    return link


def create_link_1(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 1
    :rtype: class:`graphics.graphics_robot.RotationalJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link1.stl'

    link = RotationalJoint(SE3(), stl_obj_path, scene)

    # Change color
    link.set_texture(colour=[0, 1, 0])

    return link


def create_link_2(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 2
    :rtype: class:`graphics.graphics_robot.RotationalJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link2.stl'

    link = RotationalJoint(SE3(), stl_obj_path, scene)

    # Change color
    link.set_texture(colour=[1, 0, 0])

    return link


def create_link_3(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 3
    :rtype: class:`graphics.graphics_robot.RotationalJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link3.stl'

    link = RotationalJoint(SE3(), stl_obj_path, scene)
    # stl_obj = link.get_graphic_object()

    # Change color
    link.set_texture(colour=[0, 1, 1])

    return link


def create_link_4(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 4
    :rtype: class:`graphics.graphics_robot.RotationalJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link4.stl'

    link = RotationalJoint(SE3(), stl_obj_path, scene)

    # Change color
    link.set_texture(colour=[1, 0, 1])

    return link


def create_link_5(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 5
    :rtype: class:`graphics.graphics_robot.RotationalJoint`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link5.stl'

    link = RotationalJoint(SE3(), stl_obj_path, scene)

    # Change color
    link.set_texture(colour=[1, 1, 0])

    return link


def create_link_6(scene):
    """
    Create the specific joint link and return it as a Joint object

    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Rotational joint representing joint 6
    :rtype: class:`graphics.graphics_robot.Gripper`
    """
    stl_obj_path = './roboticstoolbox/models/meshes/UNIMATE/puma560/link6.stl'

    link = Gripper(SE3(), stl_obj_path, scene)

    # Change color
    link.set_texture(colour=[0, 0, 0])

    return link
