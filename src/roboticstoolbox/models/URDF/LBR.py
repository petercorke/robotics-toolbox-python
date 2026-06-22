#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot


class LBR(Robot):
    """
    Class that imports a Kuka LBR iiwa URDF model

    ``LBR()`` is a class which imports a Kuka LBR iiwa 14 R820 robot definition
    from a URDF file.  The robot has a payload of 14 kg, and a reach of 820 mm.

    The model describes its kinematic and graphical characteristics.

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

    _urdf_path = "kuka_description/kuka_lbr_iiwa/urdf/lbr_iiwa_14_r820.xacro"
    _manufacturer = "Kuka"

    def __init__(self):

        super().__init__()

        # self.qdlim = np.array([
        #     2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.qr = np.array([0, -0.3, 0, -1.9, 0, 1.5, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = LBR()
    print(robot)
