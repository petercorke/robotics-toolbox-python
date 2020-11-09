#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class Panda(ERobot):
    """
    Class that imports a Panda URDF model

    ``Panda()`` is a class which imports a Franka-Emika Panda robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Panda()
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

        elinks, name = super().urdf_to_ets_args(
            "franka_description/robots/panda_arm_hand.urdf.xacro")

        super().__init__(
            elinks,
            name=name,
            manufacturer='Franka Emika',
            gripper_links=elinks[9]
        )

        self.qdlim = np.array([
            2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.addconfiguration("qz", np.array(
            [0, 0, 0, 0, 0, 0, 0]))

        self.addconfiguration("qr", np.array(
            [0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]))


if __name__ == '__main__':   # pragma nocover

    robot = Panda()
    print(robot)
