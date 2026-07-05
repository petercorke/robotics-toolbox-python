#!/usr/bin/env python

import re

import numpy as np
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.models.URDF.URDFRobot import URDF_read


def _patch_valkyrie_urdf(text: str) -> str:
    """Work around a broken upstream file served by robot_descriptions.

    Applies to: robot_descriptions v2.0.0's ``valkyrie_description``, which
    clones ``gkjohnson/nasa-urdf-robots`` at commit ``54cdeb1d`` and exposes
    ``val_description/model/robots/valkyrie_sim.urdf`` as a (supposedly)
    plain URDF file.

    In reality that file still contains one live, unexpanded xacro macro
    call: ``<xacro:v1_pelvis_sensors_usb .../>``. There is no definition of
    this macro anywhere in the entire cloned repo (zero ``.xacro`` files
    exist in it at all) — it's a gap in the upstream repo itself, not
    something robot_descriptions or this loader got wrong. The tag is a
    standalone, self-closing element configuring a simulated pelvis IMU
    sensor plugin (Gazebo-only, e.g. ``middle_sensor_api_tag``); it isn't
    nested inside any ``<link>``/``<joint>``, so it has no bearing on
    kinematics, dynamics, or geometry and is safe to drop entirely.

    If a future robot_descriptions/nasa-urdf-robots update actually defines
    this macro (or pre-expands it), this patch becomes a no-op (the regex
    simply won't match) — safe to remove once confirmed unnecessary.
    """
    return re.sub(r"\s*<xacro:v1_pelvis_sensors_usb\b[^>]*/>\n?", "\n", text)


class Valkyrie(Robot):
    """
    Class that imports a NASA Valkyrie URDF model

    ``Valkyrie()`` is a class which imports a NASA Valkyrie robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Valkyrie()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://github.com/gkjohnson/nasa-urdf-robots

    .. codeauthor:: Peter Corke
    """

    def __init__(self, variant="A"):

        if not variant in "ABCD":
            raise ValueError("variant must be in the range A-D")

        links, name, urdf_filepath = URDF_read("valkyrie", patch=_patch_valkyrie_urdf)

        # We wish to add an intermediate link between gripper_r_base and
        # @gripper_r_finger_r/l
        # This is because gripper_r_base contains a revolute joint which is
        # a part of the core kinematic chain and not the gripper.
        # So we wish for gripper_r_base to be part of the robot and
        # @gripper_r_finger_r/l to be in the gripper underneath a parent Link
        # gripper_r_base = links[13]
        # gripper_l_base = links[33]

        # # Find the finger links
        # r_gripper_links = [link for link in links if link.parent == gripper_r_base]
        # l_gripper_links = [link for link in links if link.parent == gripper_l_base]

        # # New intermediate links
        # r_gripper = Link(name="rightGripper", parent=gripper_r_base)
        # l_gripper = Link(name="leftGripper", parent=gripper_l_base)
        # links.append(r_gripper)
        # links.append(l_gripper)

        # # Set the finger link parent to be the new gripper base link
        # for g_link in r_gripper_links:
        #     g_link._parent = r_gripper

        # for g_link in l_gripper_links:
        #     g_link._parent = l_gripper

        super().__init__(
            links,
            name=name,
            manufacturer="NASA",
            # gripper_links=[r_gripper, l_gripper],
        )
        self._urdf_filepath = str(urdf_filepath) if urdf_filepath is not None else ""

        # self.addconfiguration_attr("qz", np.array([0, 0, 0, 0, 0, 0, 0]))
        # self.addconfiguration_attr("qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]))


if __name__ == "__main__":  # pragma nocover
    robot = Valkyrie("B")
    print(robot)
    env = robot.plot(np.zeros((robot.n,)))
    env.hold()
