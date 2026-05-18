#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Link import Link


class XYPanda(Robot):
    """
    Create model of Franka-Emika Panda manipulator on an XY platform

    xypanda = XYPanda() creates a robot object representing the Franka-Emika
    Panda robot arm mounted on an XY platform. This robot is represented using the elementary
    transform sequence (ETS).

    ETS taken from [1] based on
    https://frankaemika.github.io/docs/control_parameters.html

    :references:
        - Kinematic Derivatives using the Elementary Transform
          Sequence, J. Haviland and P. Corke

    """

    def __init__(self, workspace=5):
        """
        Create model of Franka-Emika Panda manipulator on an XY platform.

        :param workspace: workspace limits in the x and y directions, defaults to 5
        :type workspace: float, optional

        The XY part of the robot has a range -``workspace`` to ``workspace``.
        """

        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        lx = Link(
            ETS(ET.tx()), name="link-x", parent=None, qlim=[-workspace, workspace]
        )

        ly = Link(ETS(ET.ty()), name="link-y", parent=lx, qlim=[-workspace, workspace])

        l0 = Link(ET.tz(0.333) * ET.Rz(), name="link0", parent=ly)

        l1 = Link(ET.Rx(-90 * deg) * ET.Rz(), name="link1", parent=l0)

        l2 = Link(ET.Rx(90 * deg) * ET.tz(0.316) * ET.Rz(), name="link2", parent=l1)

        l3 = Link(ET.tx(0.0825) * ET.Rx(90, "deg") * ET.Rz(), name="link3", parent=l2)

        l4 = Link(
            ET.tx(-0.0825) * ET.Rx(-90, "deg") * ET.tz(0.384) * ET.Rz(),
            name="link4",
            parent=l3,
        )

        l5 = Link(ET.Rx(90, "deg") * ET.Rz(), name="link5", parent=l4)

        l6 = Link(
            ET.tx(0.088) * ET.Rx(90, "deg") * ET.tz(0.107) * ET.Rz(),
            name="link6",
            parent=l5,
        )

        ee = Link(ET.tz(tool_offset) * ET.Rz(-np.pi / 4), name="ee", parent=l6)

        elinks = [lx, ly, l0, l1, l2, l3, l4, l5, l6, ee]

        super().__init__(elinks, name="XYPanda", manufacturer="Franka Emika")

        self.qr = np.array([0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(9)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = XYPanda()
    print(robot)
