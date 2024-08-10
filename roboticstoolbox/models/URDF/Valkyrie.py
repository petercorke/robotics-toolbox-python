#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot, Link


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

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            f"val_description/model/robots/valkyrie_{variant}.urdf"
        )

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
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # self.addconfiguration_attr("qz", np.array([0, 0, 0, 0, 0, 0, 0]))
        # self.addconfiguration_attr("qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]))


if __name__ == "__main__":  # pragma nocover

    robot = Valkyrie("B")
    print(robot)
    env = robot.plot(np.zeros((robot.n,)))
    env.hold()
