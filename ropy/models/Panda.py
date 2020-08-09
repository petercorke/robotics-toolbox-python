#!/usr/bin/env python

import numpy as np
from spatialmath.base import trotz, transl
from ropy.robot.ETS import ETS
from ropy.robot.ET import ET


class Panda(ETS):
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

        et_list = [
            ET.Ttz(0.333),
            ET.TRz(),
            ET.TRx(-90*deg),
            ET.TRz(),
            ET.TRx(90*deg),
            ET.Ttz(0.316),
            ET.TRz(),
            ET.Ttx(0.0825),
            ET.TRx(90*deg),
            ET.TRz(),
            ET.Ttx(-0.0825),
            ET.TRx(-90*deg),
            ET.Ttz(0.384),
            ET.TRz(),
            ET.TRx(90*deg),
            ET.TRz(),
            ET.Ttx(0.088),
            ET.TRx(90*deg),
            ET.Ttz(0.107),
            ET.TRz(),
        ]

        tool = transl(0, 0, tool_offset) @  trotz(-np.pi/4)

        super(Panda, self).__init__(
            et_list,
            name='Panda',
            manufacturer='Franka Emika',
            tool=tool)

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0])
        self._qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
