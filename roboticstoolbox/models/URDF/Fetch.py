#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from spatialmath import SE3


class Fetch(ERobot):
    """
    Class that imports a Fetch URDF model

    ``Fetch()`` is a class which imports a Fetch robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Fetch()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "fetch_description/robots/fetch.urdf"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Fetch",
            gripper_links=links[11],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # self.grippers[0].tool = SE3(0, 0, 0.1034)

        self.qdlim = np.array(
            [4.0, 4.0, 0.1, 1.25, 1.45, 1.57, 1.52, 1.57, 2.26, 2.26]
        )

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.addconfiguration(
            "qr", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )


if __name__ == "__main__":  # pragma nocover

    robot = Fetch()
    print(robot)

    for link in robot.links:
        print(link.name)
        print(link.isjoint)
        print(len(link.collision))

    print()

    for link in robot.grippers[0].links:
        print(link.name)
        print(link.isjoint)
        print(len(link.collision))
