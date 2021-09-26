#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.robot.ELink import ELink


class XYPanda(ERobot):
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

        deg = np.pi/180
        mm = 1e-3
        tool_offset = (103)*mm

        lx = ELink(
            ETS.tx(),
            name = 'link-x',
            parent=None,
            qlim=[-workspace, workspace]
        )

        ly = ELink(
            ETS.ty(),
            name = 'link-y',
            parent=lx,
            qlim=[-workspace, workspace]
        )

        l0 = ELink(
            ETS.tz(0.333) * ETS.rz(),
            name='link0',
            parent=ly
        )

        l1 = ELink(
            ETS.rx(-90*deg) * ETS.rz(),
            name='link1',
            parent=l0
        )

        l2 = ELink(
            ETS.rx(90*deg) * ETS.tz(0.316) * ETS.rz(),
            name='link2',
            parent=l1
        )

        l3 = ELink(
            ETS.tx(0.0825) * ETS.rx(90, 'deg') * ETS.rz(),
            name='link3',
            parent=l2
        )

        l4 = ELink(
            ETS.tx(-0.0825) * ETS.rx(-90, 'deg') * ETS.tz(0.384) * ETS.rz(),
            name='link4',
            parent=l3
        )

        l5 = ELink(
            ETS.rx(90, 'deg') * ETS.rz(),
            name='link5',
            parent=l4
        )

        l6 = ELink(
            ETS.tx(0.088) * ETS.rx(90, 'deg') * ETS.tz(0.107) * ETS.rz(),
            name='link6',
            parent=l5
        )

        ee = ELink(
            ETS.tz(tool_offset) * ETS.rz(-np.pi/4),
            name='ee',
            parent=l6
        )

        elinks = [lx, ly, l0, l1, l2, l3, l4, l5, l6, ee]

        super().__init__(
            elinks,
            name='XYPanda',
            manufacturer='Franka Emika')

        self.addconfiguration(
            "qz", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration(
            "qr", np.array([0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]))


if __name__ == '__main__':   # pragma nocover

    robot = XYPanda()
    print(robot)


