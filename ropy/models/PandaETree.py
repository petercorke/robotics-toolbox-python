#!/usr/bin/env python

import numpy as np
from ropy.robot.ETS import ETS
from ropy.robot.ET import ET
from ropy.robot.ELink import ELink


class PandaETree(ETS):
    """
    Create model of Franka-Emika Panda manipulator

    panda = Panda() creates a robot object representing the Franka-Emika
    Panda robot arm. This robot is represented using the elementary
    transform sequence (ETS).

    ETS taken from [1] based on
    https://frankaemika.github.io/docs/control_parameters.html

    :references:
        - Kinematic Derivatives using the Elementary Transform
          Sequence, J. Haviland and P. Corke
    """
    def __init__(self):

        deg = np.pi/180
        mm = 1e-3
        tool_offset = (103)*mm

        l0 = ELink(
            [ET.Ttz(0.333), ET.TRz()],
            name='link0'
        )

        l1 = ELink(
            [ET.TRx(-90*deg), ET.TRz()],
            name='link1'
        )

        l2 = ELink(
            [ET.TRx(90*deg), ET.Ttz(0.316), ET.TRz()],
            name='link2'
        )

        l3 = ELink(
            [ET.Ttx(0.0825), ET.TRx(90*deg), ET.TRz()],
            name='link3'
        )

        l4 = ELink(
            [ET.Ttx(-0.0825), ET.TRx(-90*deg), ET.Ttz(0.384), ET.TRz()],
            name='link4'
        )

        l5 = ELink(
            [ET.TRx(90*deg), ET.TRz()],
            name='link5'
        )

        l6 = ELink(
            [ET.Ttx(0.088), ET.TRx(90*deg), ET.Ttz(0.107), ET.TRz()],
            name='link6'
        )

        ee = ELink(
            [ET.Ttz(tool_offset), ET.TRz(-np.pi/4)],
            name='ee'
        )

        ETlist = [l0, l1, l2, l3, l4, l5, l6, ee]

        super(PandaETree, self).__init__(
            ETlist,
            name='Panda',
            manufacturer='Franka Emika')

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
