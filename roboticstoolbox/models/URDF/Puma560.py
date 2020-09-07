#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ETS import ETS
from pathlib import Path
from math import pi


class Puma560(ETS):

    def __init__(self):

        fpath = Path('roboticstoolbox') / 'models' / 'xacro' / \
            'puma560_description' / 'urdf'
        fname = 'puma560_robot.urdf.xacro'
        abspath = fpath.absolute()

        args = super(Puma560, self).urdf_to_ets_args(
            (abspath / fname).as_posix())

        super(Puma560, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = "Unimation"
        # self.ee_link = self.ets[9]

        # zero angles, L shaped pose
        self._qz = np.array([0, 0, 0, 0, 0, 0])

        # ready pose, arm up
        self._qr = np.array([0, pi/2, -pi/2, 0, 0, 0])

        # straight and horizontal
        self._qs = np.array([0, 0, -pi/2, 0, 0, 0])

        # nominal table top picking pose
        self._qn = np.array([0, pi/4, pi, 0, pi/4, 0])

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr

    @property
    def qs(self):
        return self._qs

    @property
    def qn(self):
        return self._qn
