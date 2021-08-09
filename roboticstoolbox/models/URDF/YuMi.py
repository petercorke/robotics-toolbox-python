#!/usr/bin/env python

from roboticstoolbox.robot.ELink import ELink
import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
import spatialmath as sm


class YuMi(ERobot):
    """
    Class that imports an ABB YuMi URDF model

    ``YuMi()`` is a class which imports an ABB YuMi (IRB14000) robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.YuMi()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - `https://github.com/OrebroUniversity/yumi <https://github.com/OrebroUniversity/yumi>`_

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "yumi_description/urdf/yumi.urdf"
        )

        # We wish to add an intermediate link between gripper_r_base and
        # @gripper_r_finger_r/l
        # This is because gripper_r_base contains a revolute joint which is
        # a part of the core kinematic chain and not the gripper.
        # So we wish for gripper_r_base to be part of the robot and
        # @gripper_r_finger_r/l to be in the gripper underneath a parent ELink

        gripper_r_base = links[16]
        gripper_l_base = links[19]

        # Find the finger links
        r_gripper_links = [link for link in links if link.parent == gripper_r_base]
        l_gripper_links = [link for link in links if link.parent == gripper_l_base]

        # New intermediate links
        r_gripper = ELink(name="r_gripper", parent=gripper_l_base)
        l_gripper = ELink(name="l_gripper", parent=gripper_r_base)
        links.append(r_gripper)
        links.append(l_gripper)

        # Set the finger link parent to be the new gripper base link
        for g_link in r_gripper_links:
            g_link._parent = r_gripper

        for g_link in l_gripper_links:
            g_link._parent = l_gripper

        super().__init__(
            links,
            name=name,
            manufacturer="ABB",
            gripper_links=[r_gripper, l_gripper],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # Set the default tool transform for the end-effectors
        self.grippers[0].tool = sm.SE3.Tz(0.13)
        self.grippers[1].tool = sm.SE3.Tz(0.13)

        # self.addconfiguration("qz", np.zeros((14,)))
        # self.addconfiguration("qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]))


if __name__ == "__main__":  # pragma nocover

    robot = YuMi()
    print(robot)
