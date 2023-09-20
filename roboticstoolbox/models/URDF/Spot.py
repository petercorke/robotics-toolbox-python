#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3


class Spot(Robot):
    """
    Class that imports a Spot URDF model

    ``Spot()`` is a class which imports a Spot robot definition
    from a URDF file.  The model descroboticstoolbox/models/URDF/Spot.pyribes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Spot()
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
            "spot_description/urdf/spot_with_arm.urdf.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Boston Dynamics",
            # gripper_links=links[12],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # self.grippers[0].tool = SE3(0, 0, 0.1034)

        # self.qdlim = np.array(
        #     [4.0, 4.0, 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        # )

        # self.qr = np.array([0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(19)

        # self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    from swift import Swift

    env = Swift()
    env.launch()

    robot = Spot()
    env.add(robot)

    while True:
        print("Hello")
        break
    # print(robot)
    # q = robot.qz
    # q[12:18] = np.array([
    #     0.00043129920959472656,
    #     -3.1129815578460693,
    #     # 0.0,
    #     3.1337125301361084,
    #     1.5675101280212402,
    #     -0.0025529861450195312,
    #     -1.5732693672180176,
    # ])

    # robot.plot(q)

    # for link in robot.links:
    #     print(link.name)
    #     print(link.isjoint)
    #     print(len(link.collision))

    # print()

    # for link in robot.grippers[0].links:
    #     print(link.name)
    #     print(link.isjoint)
    #     print(len(link.collision))
