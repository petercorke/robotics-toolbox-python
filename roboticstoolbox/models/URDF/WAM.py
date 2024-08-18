#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot, ERobot2
from spatialmath import SE3


class WAM(ERobot):
    """
    Class that imports a WAM URDF model

    ``WAM()`` is a class which imports a Barrett WAM robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.WAM()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. codeauthor:: Longsen Gao
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "wam_description/barrett_model/robots/wam7.urdf.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Barrett Robtoics",
            gripper_links=links[8],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.grippers[0].tool = SE3(0, 0, 0)

        self.qdlim = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        )

        self.qr = np.array([0, 0, 0, 0, 0, 0, 0])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    r = WAM()

    r.qz

    for link in r.grippers[0].links:
        print(link)
