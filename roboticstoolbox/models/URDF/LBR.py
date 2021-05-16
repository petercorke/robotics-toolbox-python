#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class LBR(ERobot):
    """
    Class that imports a LBR URDF model

    ``LBR()`` is a class which imports a Franka-Emika LBR robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.LBR()
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
            "kuka_description/kuka_lbr_iiwa/urdf/lbr_iiwa_14_r820.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Kuka",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
            # gripper_links=elinks[9]
        )

        # self.qdlim = np.array([
        #     2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0]))

        self.addconfiguration("qr", np.array([0, -0.3, 0, -1.9, 0, 1.5, np.pi / 4]))


if __name__ == "__main__":  # pragma nocover

    robot = LBR()
    print(robot)
