#!/usr/bin/env python

import numpy as np
import os
from ropy.robot.ETS import ETS
from pathlib import Path

import spatialmath as sm


class UR5(ETS):

    def __init__(self):

        fpath = Path('ropy/models/xarco/ur/urdf/ur5_joint_limited_robot.urdf.xacro')
        abspath = os.getcwd() + '/ropy/models/xarco/ur/urdf/'

        args = super(UR5, self).urdf_to_ets_args(fpath)

        super(UR5, self).__init__(
            args[0],
            name=args[1])

        self.manufacturer = 'Universal Robotics'
        # self.ee_link = self.ets[9]

        self._qz = np.array([0, 0, 0, 0, 0, 0])
        self._qr = np.array([np.pi, 0, 0, 0, np.pi/2, 0])

        for link in self.ets:
            for gi in link.geometry:
                if gi.filename[0] != '/':
                    gi.filename = abspath + gi.filename

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr
