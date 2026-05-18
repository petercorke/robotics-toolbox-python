#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.Robot import Robot


class Mico(Robot):
    """
    Class that imports a Mico URDF model

    ``Panda()`` is a class which imports a Kinova Mico robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Mico()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "kinova_description/urdf/j2n4s300_standalone.xacro"
        )

        gripper_base = links[6]

        # Find the finger links
        gripper_links = [link for link in links if link.parent == gripper_base]

        # New intermediate link
        gripper = Link(name="gripper", parent=gripper_base)
        links.append(gripper)

        # Set the finger link parent to be the new gripper base link
        for g_link in gripper_links:
            g_link._parent = gripper

        super().__init__(
            links,
            name=name,
            manufacturer="Kinova",
            gripper_links=[gripper],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qr = np.array([0, 45, 60, 0]) * np.pi / 180
        self.qz = np.zeros(4)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = Mico()
    print(robot)
